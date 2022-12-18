import torch.optim as optim
from omegaconf import DictConfig


def get_scheduler(cfg: DictConfig, optimizer: optim.Optimizer):
    if cfg.scheduler.name == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.CosineAnnealingWarmRestarts.T_0,
            eta_min=cfg.CosineAnnealingWarmRestarts.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler.name}")