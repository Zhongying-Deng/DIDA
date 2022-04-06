"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import math
import torch

AVAI_SCHEDS = ['single_step', 'multi_step', 'cosine', 'fixmatch']


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            'Unsupported scheduler: {}. Must be one of {}'.format(
                lr_scheduler, AVAI_SCHEDS
            )
        )

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    elif lr_scheduler == 'fixmatch':
        iteration = optim_cfg.ITER_PER_EPOCH
        scheduler = get_cosine_schedule_with_warmup(optimizer, float(max_epoch) * iteration)

    return scheduler


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_warmup_steps=0,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)

