import torch
import torch.nn as nn
import torch.nn.functional as F
from wene.transformer.norm import RMSNorm
from wenet.transformer.attention import RopeMultiHeadedAttention
from wenet.transformer.positionwise_feed_forward import GatedVariantsMLP


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

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, nonlinearity='linear')  # lecun_normal approx
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, emb):
        # emb shape: (B, D)
        emb = F.gelu(emb)
        emb = self.lin(emb)  # (B, 3*D)
        shift_msa, scale_msa, gate_msa = torch.chunk(emb, 3,
                                                     dim=-1)  # Each: (B, D)

        # Expand to match x shape: (B, T, D)
        shift_msa = shift_msa.unsqueeze(1)  # (B, 1, D)
        scale_msa = scale_msa.unsqueeze(1)  # (B, 1, D)
        gate_msa = gate_msa.unsqueeze(1)  # (B, 1, D)

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
