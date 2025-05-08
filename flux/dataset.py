import json
from functools import partial

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from wenet.dataset.datapipes import WenetRawDatasetSource


def decode_wav(sample):
    obj = json.loads(sample['line'])
    filepath = obj['wav']
    speech_token = obj['speech_token']
    audio, sample_rate = torchaudio.load(filepath)
    return {'wav': audio, "sample_rate": sample_rate, "token": speech_token}


def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def filter_by_length(sample, max_seconds=30, min_seconds=0.5):
    wav = sample['wav']
    sr = sample['sample_rate']
    duration = wav.shape[1] / sr
    if duration <= max_seconds and duration >= min_seconds:
        return True
    return False


def sort_by_feats(sample):
    assert 'wav' in sample
    assert isinstance(sample['wav'], torch.Tensor)
    return sample['wav'].size(1)


def compute_mels(sample, mel_fn):
    wav = sample['wav']
    padding = torch.zeros(1, wav.shape[1])
    mel, out_padding = mel_fn(wav, padding)  # [1, C, T]
    out_padding = 1 - out_padding.to(torch.int64)
    mel = mel[:, :, :out_padding.sum(-1)[0]]
    mel = mel.transpose(1, 2)
    sample['mel'] = mel
    return sample


def padding(data, pad_value=0):

    double_token_lst = []
    mels_lst = []

    tokens = [sample['token'] for sample in data]
    mels = [sample['mel'] for sample in data]
    for (token, mel) in zip(tokens, mels):
        doubled_token = [x for item in token for x in (item, item)]
        min_len = min(len(doubled_token), mel.shape[1])
        double_token_lst.append(
            torch.tensor(doubled_token[:min_len], dtype=torch.int64))
        mels_lst.append(mel[0][:min_len, :])

    mels_lens = [sample['mel'].shape[0] for sample in mels_lst]

    speech_tokens = pad_sequence(double_token_lst,
                                 batch_first=True,
                                 padding_value=pad_value)
    mels = pad_sequence(mels_lst, batch_first=True, padding_value=pad_value)
    mels_lens = torch.tensor(mels_lens, dtype=torch.int64)
    return {
        'mels': mels,
        'mels_lens': mels_lens,
        'speech_tokens': speech_tokens,
    }


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'mel' in sample
        assert isinstance(sample['mel'], torch.Tensor)
        new_sample_frames = sample['mel'].size(1)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False


def init_dataset_and_dataloader(files,
                                batch_size,
                                num_workers,
                                prefetch,
                                shuffle,
                                steps,
                                mel_fn,
                                drop_last=False,
                                sample_rate=24000,
                                seed=2025,
                                sort_buffer_size=1024,
                                shuffle_size=2048,
                                batch_type='static',
                                split='train'):

    dataset = WenetRawDatasetSource(files,
                                    cycle=steps,
                                    shuffle=shuffle,
                                    partition=True)
    dataset = dataset.shuffle(buffer_size=shuffle_size)
    dataset = dataset.map(decode_wav)
    dataset = dataset.filter(filter_by_length)
    if split == 'train':
        dataset = dataset.sort(buffer_size=sort_buffer_size,
                               key_func=sort_by_feats)
    dataset = dataset.map(partial(resample, resample_rate=sample_rate))
    dataset = dataset.map(partial(compute_mels, mel_fn=mel_fn))
    assert batch_type in ['static', 'dynamic']
    if batch_type == 'static':
        dataset = dataset.batch(batch_size,
                                wrapper_class=partial(padding, pad_value=0.0),
                                drop_last=drop_last)
    else:
        max_frames_in_batch = batch_size
        dataset = dataset.dynamic_batch(
            DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=partial(padding, pad_value=0.0),
        )
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            persistent_workers=True,
                            prefetch_factor=prefetch,
                            generator=generator)
    return dataset, dataloader
