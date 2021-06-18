import os
import sys
import math
import glob
import shutil
import random
import tempfile
import importlib
from pathlib import Path

import librosa
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

import hubconf
from optimizers import get_optimizer
from schedulers import get_scheduler
from utility.helper import is_leader_process, count_parameters, get_model_state, show, defaultdict

#SAMPLE_RATE = 16000
TRIM_DB_MAX = 60
TRIM_FRAME_LEN = 2048
TRIM_HOP_LEN = 512
TRIM_PAD = 512

class CovidDetection():
    
    def __init__(self):

        self.init_ckpt = torch.load('./result/downstream/exp-1/states-218000.ckpt', map_location='cpu')
        self.init_ckpt_upstream = torch.load('./result/pretrain/covidModel-3/states-epoch-1145.ckpt', map_location='cpu')

        self.config = self.init_ckpt['Config']
        self.args = self.init_ckpt['Args']

        self.args.upstream_ckpt = './result/pretrain/covidModel-3/states-epoch-1145.ckpt'
        self.args.init_ckpt = './result/downstream/exp-1/states-218000.ckpt'
        self.args.upstream_model_config = './result/pretrain/covidModel-3/options.yaml'
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.upstream = self._get_upstream()
        self.upstream.eval()
        self.downstream = self._get_downstream()
        self.downstream.eval()

        # init the cached position encoding table
        zeros = [0 for i in range(221)]
        self.predict_wavs([zeros])


    def _get_upstream(self):
        Upstream = getattr(hubconf, self.args.upstream)
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        upstream = Upstream(
            feature_selection = self.args.upstream_feature_selection,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
            ckpt = self.args.upstream_ckpt,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        interface_fn = ['get_output_dim', 'get_downsample_rate']
        for fn in interface_fn:
            assert hasattr(upstream, fn)

        if self.args.verbose:
            show(f'[Runner] - Upstream model architecture: {upstream}')
            show(f'[Runner] - Upstream has {count_parameters(upstream)} parameters')
            show(f'[Runner] - Upstream output dimension: {upstream.get_output_dim()}')
            downsample = upstream.get_downsample_rate()
            show(f'[Runner] - Upstream downsample rate: {downsample} ({downsample / SAMPLE_RATE * 1000} ms/frame)')

        init_upstream = self.init_ckpt.get('Upstream')
        if init_upstream:
            show('[Runner] - Loading upstream weights from the previous experiment')
            upstream.load_state_dict(init_upstream)

        #if is_initialized() and self.args.upstream_trainable:
        #    upstream = DDP(upstream, device_ids=[self.args.local_rank], find_unused_parameters=True)
        #    for fn in interface_fn:
        #        setattr(upstream, fn, getattr(upstream.module, fn))

        return upstream


    def _get_downstream(self):
        module_path = f'downstream.{self.args.downstream}.cough_predict'
        Downstream = getattr(importlib.import_module(module_path), 'CoughPredict')
        downstream = Downstream(
            upstream_dim = self.upstream.get_output_dim(),
            upstream_rate = self.upstream.get_downsample_rate(),
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        if self.args.verbose:
            show(f'[Runner] - Downstream model architecture: {downstream}')
            show(f'[Runner] - Downstream has {count_parameters(downstream)} parameters')


        init_downstream = self.init_ckpt.get('Downstream')
        if init_downstream:
            show('[Runner] - Loading downstream weights from the previous experiment')
            downstream.load_state_dict(init_downstream)

        #print(is_initialized())
        #if is_initialized():
        #    downstream = DDP(downstream, device_ids=[self.args.local_rank], find_unused_parameters=True)
        #    for fn in interface_fn:
        #        setattr(downstream, fn, getattr(downstream.module, fn))

        return downstream

    def predict_wavs(self, wavs):

        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
        with torch.no_grad():
            features = self.upstream(wavs)
            predicted_class_id = self.downstream(features)

        return predicted_class_id

    def split_and_batch_audio(self, audio_file_path):
    
        if False == os.path.isfile(audio_file_path):
            print(f'Audio file {audio_file_path} does not exist')
            sys.exit(1)
        
        x , sr = librosa.load(audio_file_path, sr = None)
        ints = librosa.effects.split(x, top_db = TRIM_DB_MAX, frame_length = TRIM_FRAME_LEN, hop_length = TRIM_HOP_LEN)
     
        wavs = []
        x_trimmed = [] 
        for ind in range(ints.shape[0]):
        
            start = int(ints[ind][0])
            end = int(ints[ind][1])
            if (end - start <= 1024):
                continue
            wavs.append(x[start:end])
            x_trimmed = np.hstack((x_trimmed,x[start:end]))
        
        return wavs, x_trimmed


def predict(metadata_df:pd.DataFrame):

    cd = CovidDetection()

    def predict_helper(audio_path):
        
        wavs, wav_trimmed = cd.split_and_batch_audio(audio_path)
        if len(wav_trimmed) > 2e6:
            return 'too long'

        wavs = [torch.FloatTensor(wav).to(cd.args.device) for wav in wavs]
        with torch.no_grad():
            features = cd.upstream(wavs)
            predicted_class_id = cd.downstream(features)

        predicted_class_id = predicted_class_id.float().mean().to('cpu').numpy()
        predicted_class_id = 1 if predicted_class_id >= 0.5 else 0
        print(audio_path+':', predicted_class_id)
        return predicted_class_id

    predict_df = metadata_df[['audio_path']]
    predict_df['pcr_test_result_predicted'] = predict_df['audio_path'].apply(predict_helper)
    return predict_df



if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: predict.py  metadata_csv_filepath")
        sys.exit(1)

    metadata_csv_filepath = sys.argv[1]
    data_dir_name = metadata_csv_filepath.split('.')[-2]
 

    metadata_df = pd.read_csv(metadata_csv_filepath)
    metadata_df['audio_path'] = data_dir_name + '/' + metadata_df['audio_path']
    metadata_df = predict(metadata_df)

    print(metadata_df['pcr_test_result_predicted'])
    output_dir_name = data_dir_name + '-predictd.csv' 
    metadata_df.to_csv(output_dir_name)

