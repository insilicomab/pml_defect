import torch
import torch.nn as nn
import timm
import pytorch_metric_learning
from pytorch_metric_learning.utils import common_functions


class ConvnextBase(nn.Module):
    def __init__(self):
        super(ConvnextBase, self).__init__()
        self.model_name = 'convnext_base'
        self.pretrained = True

        self.trunk = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.trunk.head.fc = common_functions.Identity()
        self.embedder = nn.Linear(1024, 512)
    
    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        return x