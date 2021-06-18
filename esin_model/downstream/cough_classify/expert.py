# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
#-------------#
from downstream.model import *
from .dataset import CoughClassifiDataset
from argparse import Namespace
from pathlib import Path


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert

        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = CoughClassifiDataset('train', root_dir, self.datarc['train_batch_size'], 
                self.datarc['max_timestep'], self.datarc['class_weight'])
        self.test_dataset = CoughClassifiDataset('test', root_dir, self.datarc['train_batch_size'])
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = self.train_dataset.speaker_num,
            **model_conf,
        )
        tensor = torch.tensor((), dtype=torch.float32)
        #class_weights = torch.FloatTensor(self.datarc['class_weight'])
        #self.objective = nn.CrossEntropyLoss(weight=class_weights)
        #self.objective = nn.CrossEntropyLoss(weight=tensor.new_tensor([0.505, 0.495]))
        self.objective = nn.CrossEntropyLoss()
        
        self.logging = os.path.join(expdir, 'log.log')
        self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset):

        print("&&&&&&&&&&&&&")
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            sampler = dataset.sampler,  
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        def multi_acc(pred, label):
            accs_per_label_pct = []
            tags = torch.argmax(pred, dim=1)
            for c in range(2):  # the three classes
                of_c = label == c
                num_total_per_label = of_c.sum()
                of_c &= tags == label
                num_corrects_per_label = of_c.sum()
                accs_per_label_pct.append(num_corrects_per_label / num_total_per_label * 100)
            return accs_per_label_pct

        per_class_acc = multi_acc(predicted, labels)
        per_class_acc_0 = per_class_acc[0].cpu().float()
        per_class_acc_1 = per_class_acc[1].cpu().float()
        if not torch.isnan(per_class_acc_0):
            records['acc/class 0'].append(per_class_acc_0)
        if not torch.isnan(per_class_acc_1):
            records['acc/class 1'].append(per_class_acc_1)
        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'voxceleb1/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(self.logging, 'a') as f:
                if key == 'acc':
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
        return save_names
