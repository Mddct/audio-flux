import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from wene.transformer.norm import RMSNorm
from wenet.transformer.attention import RopeMultiHeadedAttention
from wenet.transformer.embeddding import RopePositionalEncoding
from wenet.transformer.positionwise_feed_forward import GatedVariantsMLP


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:

    t = timesteps[:, None, None]
    noisy_samples = t * noise + (1 - t) * original_samples
    return noisy_samples


class TimestepEmbedding(nn.Module):

    def __init__(self, in_size, embedding_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_size, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        return x


class CombinedTimestepSpeechTokenGuidanceEmbeddings(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.timestep_embedding = TimestepEmbedding(embedding_dim,
                                                    embedding_dim)
        self.gudience_embedding = TimestepEmbedding(embedding_dim,
                                                    embedding_dim)

    def forward(self, timestep, guidance):
        timestep_emb = self.timestep_embedding(timestep.to(
            guidance.dtype)).unsqueeze(1)  # [B,1,D]
        guidance_emb = self.guidance_embedding(guidance.to(guidance.dtype))
        conditioning = timestep_emb + guidance_emb

        return conditioning


class AdaLayerNormZeroSingle(nn.Module):
    """
    Adaptive LayerNorm Zero (adaLN-Zero) module.

    Args:
        embedding_dim (int): The size of each embedding vector.
        norm_type (str): Type of normalization to use. Currently only 'layer_norm' is supported.
        bias (bool): Whether to include bias in the linear layer.
        dtype (torch.dtype): Data type for parameters and computation.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.norm_type = norm_type
        self.bias = bias

        self.lin = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self._init_weights(self.lin)

        self.norm = nn.LayerNorm(embedding_dim,
                                 elementwise_affine=False,
                                 bias=False,
                                 eps=1e-6)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, nonlinearity='linear')  # lecun_normal approx
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, emb):
        # emb shape: (B, T, D)
        emb = F.gelu(emb)
        emb = self.lin(emb)  # (B, 3*D)
        shift_msa, scale_msa, gate_msa = torch.chunk(emb, 3,
                                                     dim=-1)  # Each: (B, D)

        if self.norm_type == "layer_norm":
            x = self.norm(x) * (1 + scale_msa) + shift_msa
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({self.norm_type}) provided. Supported ones are: 'layer_norm'."
            )

        return x, gate_msa


class FluxSingleTransformerBlock(nn.Module):
    """
    A Transformer block following the MMDiT architecture (Stable Diffusion 3).
    Modified only pre nrom

    Args:
        dim (int): The number of channels in the input and output.
        num_attention_heads (int): The number of heads to use for multi-head attention.
        attention_head_dim (int): The number of channels in each head.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config

        self.norm1 = AdaLayerNormZeroSingle(config.output_size)
        self.norm2 = RMSNorm(config.output_size, config.norm_eps)

        # Replace with actual attention implementation
        encoder_selfattn_layer_args = (
            config.attention_heads,
            config.output_size,
            config.attention_dropout_rate,
            config.query_bias,
            config.key_bias,
            config.value_bias,
            config.use_sdpa,
            config.n_kv_head,
            config.head_dim,
        )
        self.attn = RopeMultiHeadedAttention(*encoder_selfattn_layer_args)
        self.mlp = GatedVariantsMLP(
            config.output_size,
            config.linear_units,
            dropout_rate=0.0,
            activation=torch.nn.GELU(),
            bias=False,
        )

    def forward(self, x, temb, mask, pos_emb):
        residual = x

        norm_hidden_states, gate = self.norm(x, temb)  # (B, T, D)

        x_att, _ = self.attn(norm_hidden_states, norm_hidden_states,
                             norm_hidden_states, mask, pos_emb)

        hidden_states = residual + gate.unsqueeze(1) * x_att
        norm_hidden_states = self.norm2(hidden_states)

        hidden_states = self.mlp(norm_hidden_states) + norm_hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class AudioTransformer2DModel(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.blocks = torch.nn.ModuleList([
            FluxSingleTransformerBlock(config)
            for _ in range(config.num_blocks)
        ])

        self.pos_emb = RopePositionalEncoding(config.output_size,
                                              config.head_dim,
                                              dropout_rate=0.0,
                                              max_len=25 * 4 * 30 * 2)
        self.speech_token_embed = torch.nn.Embedding(config.speech_vocab_size,
                                                     config.output_size)

        self.time_speech_embed = CombinedTimestepSpeechTokenGuidanceEmbeddings(
            config.output_size)

        self.lin = torch.nn.Linear(config.n_mels, config.output_size)

    def timestep_embedding(self,
                           t: torch.Tensor,
                           dim: int,
                           max_period: float = 10000,
                           time_factor: float = 1000.0) -> torch.Tensor:
        """
        Generate sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): shape (batch,), can be float.
            dim (int): embedding dimension.
            max_period (float): controls the minimum frequency of the embeddings.
            time_factor (float): scales the input timestep (default 1000.0)

        Returns:
            torch.Tensor: shape (batch, dim), timestep embeddings.
        """

        t = t * time_factor  # Scale time
        half = dim // 2

        device = t.device
        dtype = t.dtype

        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(0, half, dtype=torch.float32, device=device) /
            half).to(dtype)  # shape: (half,)

        args = t[:, None] * freqs[None, :]  # shape: (batch, half)
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)],
            dim=-1)  # shape: (batch, dim or dim - 1)

        if dim % 2 != 0:
            # Pad with zeros if dim is odd
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, batch: dict):
        """forward for training
        """
        speech_tokens = batch['speech_tokens']
        speech_tokens_lens = batch['speech_tokens_lens']
        speech_embeds = self.speech_token_embed(speech_tokens)

        mels = batch['mel']
        mels_lens = batch['mels_lens']

        timestep = batch['timestep']
        timestep = self.timestep_embedding(timestep, 256)  # [B, T]

        temb = self.time_speech_embed(timestep)
        hiden_states = self.lin(mels)

        # TODO: cfg or soundstorm mask
        hidden_states = temb + hiden_states
        # TODO: checkpointing
        for layer in self.blocks:
            hidden_states = layer(hidden_states, temb)
        return hidden_states
