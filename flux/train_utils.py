import os

import torch
import torch.distributed as dist
import torch.optim as optim
import torchaudio
from absl import logging
from torch.utils.tensorboard import SummaryWriter
from wenet.utils.mask import make_non_pad_mask

from flux.dataset import init_dataset_and_dataloader
from flux.utils import get_cosine_schedule_with_warmup, init_distributed
from model import DITModel, add_noise


def samples_sequence_mask(seq_lens: torch.Tensor, max_lens):
    B = seq_lens.shape[0]
    device = seq_lens.device
    random_floats = torch.rand(B, device=device)  # Shape [B,]
    scaled_random_floats = random_floats * seq_lens.float()
    s = torch.floor(scaled_random_floats).to(torch.int64)
    time_indices = torch.arange(max_lens, dtype=s.dtype,
                                device=device)  # Shape [T,]

    s_expanded = s.unsqueeze(1)  # Shape [B, 1]

    mask = s_expanded > time_indices  # Shape [B, T], dtype=torch.bool
    return mask


class TrainState:

    def __init__(
        self,
        config,
    ):

        _, _, self.rank = init_distributed(config)
        model = DITModel(config)
        model.cuda()
        self.config = config
        # TODO: FSDP V2
        self.model = torch.nn.parallel.DistributedDataParallel(model)
        self.device = config.device

        self.sample_rate = config.sample_rate
        self.learning_rate = config.learning_rate
        self.warmup_steps = config.warmup_steps

        self.max_steps = config.max_train_steps

        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_size,
            n_mels=config.n_mels,
            center=False,
            power=1.0,
            f_min=0,
            f_max=None,
            norm='slaney',
            mel_scale='slaney',
        )
        _, self.dataloader = init_dataset_and_dataloader(
            config.train_data,
            config.per_device_batch_size,
            config.num_workers,
            config.prefetch,
            True,
            self.max_steps,
            sample_rate=config.sample_rate,
            seed=config.seed,
            mel_fn=mel_fn)
        # self.evaluate_utmos = config.evaluate_utmos
        # self.evaluate_pesq = config.evaluate_pesq
        # self.evaluate_periodicty = config.evaluate_periodicty

        # TODO: user clu async torch writer
        self.writer = SummaryWriter(config.tensorboard_dir)

        self.timesteps = torch.linspace(1, 0, steps=50 + 1).cuda()
        # Optimizers
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        # Schedulers
        self.scheduler = get_cosine_schedule_with_warmup(
            self.opt, self.warmup_steps, self.max_steps // 2)
        self.step = 0

    def train_step(self, batch, device):
        mels, mels_lens = batch['mels'].to(device), batch['mels_lens'].to(
            device)
        speech_tokens = batch['speech_tokens'].to(device)

        timesteps = torch.randint(
            low=0,
            high=len(self.timesteps) - 1,
            size=(mels.shape[0], ),
            device=mels.device,
        )
        noise = torch.randn(mels.shape, dtype=mels.dtype, device=mels.device)
        noise_mels = add_noise(mels, noise, self.timesteps[timesteps])
        # NOTE(Mddct):
        #     1 condition tokens never  mask
        #     2 mel nver mask before t
        t_mask = samples_sequence_mask(mels_lens, mels_lens.max())
        noise_mels = torch.where(t_mask.unsqueeze(-1), mels, noise_mels)

        model_pred, mask = self.model(speech_tokens, noise_mels, mels_lens,
                                      timesteps)
        target = noise - mels
        loss = (target - model_pred)**2
        l_mask = mask.transpose(1,
                                2) * (1 - t_mask.to(torch.int64)).unsqueeze(-1)
        masked_sum = (loss * l_mask).sum()
        num_valid = l_mask.sum()
        loss_mean = masked_sum / (num_valid + 1e-7)

        loss_mean.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.config.clip_grad_norm)

        log_str = f'[RANK {self.rank}] step_{self.step+1}: '
        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()
        if self.rank == 0:
            self.writer.add_scalar("train/mel_loss", loss_mean, self.step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.step)

        opt_lrs = [group['lr'] for group in self.opt.param_groups]
        log_str += f"loss: {loss_mean.item()}\tgrad_norm: {grad_norm.item()}\t"
        for i, lr in enumerate(opt_lrs):
            if self.rank == 0:
                self.writer.add_scalar('train/lr_{}'.format(i), lr, self.step)
            log_str += f' lr_{i}: {lr:>6.5f}'

        if (self.step + 1) % self.config.log_interval == 0:
            logging.info(log_str)

    def train(self):
        if self.config.checkpoint != '':
            self.resume(self.config.checkpoint)
        self.model.train()
        for batch in self.dataloader:
            dist.barrier()
            self.train_step(batch, self.config.device)
            if (self.step + 1) % self.config.checkpoint_every_steps == 0:
                self.save()
            self.step += 1
            if self.step >= self.max_steps:
                print("Training complete.")
                return

    def save(self):
        if self.rank == 0:
            checkpoint_dir = os.path.join(self.config.model_dir,
                                          self.config.run_name,
                                          f'step_{self.step}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            model_state_dict = self.model.module.state_dict()
            meta = {
                'model': model_state_dict,
                'step': self.step,
            }
            torch.save(meta, os.path.join(checkpoint_dir, 'model.pt'))
            opt_state_dict = self.opt.state_dict()
            torch.save(opt_state_dict, os.path.join(checkpoint_dir, 'opt.pt'))
            logging.info(
                f'[RANK {self.rank}] Checkpoint: save to checkpoint {checkpoint_dir}'
            )

    def resume(self, checkpoint_dir: str):

        model = self.model.module
        ckpt = torch.load(os.path.join(checkpoint_dir, 'model.pt'),
                          map_location='cpu',
                          mmap=True)
        model.load_state_dict(ckpt['model'])
        self.step = ckpt['step'] + 1  # train from new step

        opt_disc = self.opt
        ckpt = torch.load(os.path.join(checkpoint_dir, 'opt.pt'),
                          map_location='cpu',
                          mmap=True)
        opt_disc.load_state_dict(ckpt)

        logging.info(
            f'[RANK {self.rank}] Checkpoint: load  checkpoint {checkpoint_dir}'
        )
        dist.barrier()

        self.scheduler.set_step(self.step)
