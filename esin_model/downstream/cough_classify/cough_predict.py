import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from downstream.model import *
from .dataset import CoughClassifiDataset
from argparse import Namespace
from pathlib import Path


class CoughPredict(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(CoughPredict, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.register_buffer('best_score', torch.zeros(1))

        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        root_dir = Path(self.datarc['file_path'])

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = 2,
            **model_conf,
        )
        tensor = torch.tensor((), dtype=torch.float32)

    def forward(self, features):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)


        predicted_classid = predicted.max(dim=-1).indices
        return predicted_classid
