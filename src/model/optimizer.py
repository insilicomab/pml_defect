import torch.optim as optim
import torch.nn as nn
from omegaconf import DictConfig


def get_optimizer(cfg: DictConfig, net: nn.Module):
    if cfg.optimizer.name == 'Adam':
        return optim.Adam(
            net.parameters(),
            lr=cfg.optimizer.adam.lr,
            weight_decay=cfg.optimizer.adam.weight_decay
        )
    elif cfg.optimizer.name == 'SGD':
        return optim.SGD(
            net.parameters(),
            lr=cfg.optimizer.sgd.lr,
            weight_decay=cfg.optimizer.sgd.weight_decay
        )
    else:
        raise ValueError(f'Unknown optimizer: {cfg.optimizer.name}')