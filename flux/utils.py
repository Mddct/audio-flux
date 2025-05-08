import math
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR


def init_distributed(configs):

    local_rank = os.environ.get('LOCAL_RANK', 0)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    print('training on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))

    return world_size, local_rank, rank


def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *,
                                               num_warmup_steps: int,
                                               num_training_steps: int,
                                               num_cycles: float):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return max(
        0.0,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
