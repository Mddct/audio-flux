from typing import Dict

import s3tokenizer
import torch
from absl import app, flags
from ml_collections import config_flags

from flux.model import DITModel

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config')

flags.DEFINE_string('wav', None, help='audio file', required=True)
flags.DEFINE_string('checkpoint', None, help='model checkpoint', required=True)
flags.DEFINE_integer('num_inference_steps',
                     None,
                     help='dit inference steps',
                     required=True)


def main(_):

    config = FLAGS.config
    print(config)

    model = DITModel(config)

    tokenizer = s3tokenizer.load_model("speech_tokenizer_v1")
    mels = []
    wav_paths = [FLAGS.wav]
    for wav_path in wav_paths:
        audio = s3tokenizer.load_audio(wav_path)
        mels.append(s3tokenizer.log_mel_spectrogram(audio))
    mels, mels_lens = s3tokenizer.padding(mels)
    speech_tokens, speech_tokens_lens = tokenizer.quantize(
        mels.cuda(), mels_lens.cuda())

    seed = 2025
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(1, speech_tokens.shape[1], config.n_mels, generator=g)
    timesteps = torch.linspace(1, 0, config.num_inference_steps + 1)
    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]

    latents = noise
    for step in range(config.inference_steps):
        t_curr = c_ts[step]
        t_prev = p_ts[step]
        t_vec = torch.full((noise.shape[0], ), t_curr, dtype=noise.dtype)
        pred = model(speech_tokens, latents, speech_tokens_lens, t_vec)
        latents = latents + (t_prev - t_curr) * pred
    mels = latents
    print(mels.shape)


if __name__ == '__main__':
    app.run(main)
