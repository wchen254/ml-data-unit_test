import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
import pandas

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


class CoughClassifiDataset(Dataset):
    def __init__(self, mode, file_path, batch_size, max_timestep=None, class_weights=None):

        self.root = file_path
        self.speaker_num = 2
        self.max_timestep = max_timestep
        self.class_weights = class_weights
        self.batch_size = batch_size

        cache_path = os.path.join(CACHE_PATH, f'{mode}.pkl')
        if os.path.isfile(cache_path):
            print(f'[CoughClassifiDataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                db_data_label = pickle.load(cache)
                dataset = db_data_label['file_path']
                label = db_data_label['label']
        else:
            dataset, label = eval("self.{}".format(mode))()
            db_data_label = {'file_path' : dataset, 'label' : label}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(db_data_label, cache)
        print(f'[CoughClassifiDataset] - there are {len(dataset)} files found')

        self.dataset = dataset
        self.label = label

        #self.class_weights = [0.1, 0.9]
        #label_weights = [self.class_weights[x] for x in label]
        #label_weights = torch.FloatTensor(label_weights)
        #self.sampler = WeightedRandomSampler(weights = label_weights, num_samples = len(label_weights), replacement=True)
        self.sampler = None

    
    def train(self):

        df = pandas.read_csv("/tf/audio-models/s3prl/downstream/cough_classify/coughvid_train.csv")
        df = df.sample(frac=1)
        df = df.sample(frac=1)
        df = df.sample(frac=1)
        labels = [0 if x == 'healthy'  else 1 for x in df['labels']]

        return df['file_paths'], labels
         
    def test(self):

        df = pandas.read_csv("/tf/audio-models/s3prl/downstream/cough_classify/coughvid_test.csv")
        labels = [0 if x == 'healthy'  else 1 for x in df['labels']]
        return df['file_paths'], labels

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep
  
        return wav.numpy(), self.label[idx]
        
    def collate_fn(self, samples):        
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
