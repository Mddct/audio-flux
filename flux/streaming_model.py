import torch
from wenet.utils.mask import causal_or_lookahead_mask, make_non_pad_mask

from flux.model import DITModel


class StreamingDITmodel(DITModel):

    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, speech_tokens: torch.Tensor, mels: torch.Tensor,
                mels_lens: torch.Tensor, timesteps: torch.Tensor):
        """forward for training
        """
        speech_embeds = self.speech_token_embed(speech_tokens)

        mask = make_non_pad_mask(mels_lens).unsqueeze(1)
        att_mask = causal_or_lookahead_mask(mask, self.config.right_context,
                                            self.config.left_context)

        timesteps = self.timestep_embedding(timesteps,
                                            self.config.output_size)  # [B, D]
        temb = self.time_speech_embed(timesteps, speech_embeds)

        hidden_states = self.lin(mels)
        hidden_states = temb + hidden_states
        # TODO: checkpointing
        hidden_states, pos_emb = self.pos_emb(hidden_states)
        for layer in self.blocks:
            hidden_states = layer(hidden_states, temb, att_mask, pos_emb)

        out = self.out(self.after_norm(hidden_states))
        return out, mask
