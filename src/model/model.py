import torch
import torch.nn as nn
import timm
import pytorch_metric_learning
from pytorch_metric_learning.utils import common_functions


class ConvnextBase(nn.Module):
    def __init__(self, pretrained, embedding_size):
        super(ConvnextBase, self).__init__()
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        self.trunk = timm.create_model(
            'convnext_base',
            pretrained=self.pretrained,
        )
        self.trunk.head.fc = common_functions.Identity()
        self.embedder = nn.Linear(1024, self.embedding_size)
    
    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        return x


def get_model(model_name, pretrained, embedding_size):
    if model_name=='convnext_base':
        return ConvnextBase(
            pretrained=pretrained,
            embedding_size=embedding_size,
        )