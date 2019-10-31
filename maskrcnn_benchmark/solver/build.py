# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        # _C.SOLVER.BASE_LR = 0.001
        lr = cfg.SOLVER.BASE_LR
        # _C.SOLVER.WEIGHT_DECAY = 0.0005
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            # _C.SOLVER.BIAS_LR_FACTOR = 2
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # _C.SOLVER.MOMENTUM = 0.9
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

# 调节lr，用上了预热学习率
def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS, # _C.SOLVER.STEPS = (30000,)
        cfg.SOLVER.GAMMA, # _C.SOLVER.GAMMA = 0.1
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,  # _C.SOLVER.WARMUP_FACTOR = 1.0 / 3
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,  # _C.SOLVER.WARMUP_ITERS = 500
        warmup_method=cfg.SOLVER.WARMUP_METHOD,  # _C.SOLVER.WARMUP_METHOD = "linear"
    )
