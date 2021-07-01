from joblib import dump, load
import pandas as pd
import os
import librosa
import numpy as np
from typing import Tuple
from scipy import signal
import emd
import sklearn
import dask.dataframe as dd
import dask.dataframe.utils as daskutils
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get
from multiprocessing import Process, freeze_support
import subprocess

##################
#
##################
def clipping_detect(audio: np.ndarray, fs: int, fn: str, threshold=0.3) -> float:
    # rate, data = wavfile.read(file)
    length = len(audio)
    # Nomalize data (max value = 2^15) and mirror negative values in x-axis
    data_norm = abs(audio/max(audio))
    # Peak Detection
    peaks = signal.find_peaks(data_norm, height=0.1, distance=25, prominence=0.1)[0]
    # First Derivative
    fderiv = np.diff(data_norm)
    clipped_peaks = []
    # Detects if there is flatness in the waveform on either side of the peak
    for peak in peaks:
        # Ignore peaks too close to the edge of the file
        if peak>4 and peak<(length-4):
            left_slop = (fderiv[peak-1]+fderiv[peak-2]+fderiv[peak-3])/3
            right_slop = (fderiv[peak+1]+fderiv[peak+2]+fderiv[peak+3])/3
            if (left_slop < 0.004) or (right_slop > -0.004):
                clipped_peaks.append(peak)
    num_clipped = len(clipped_peaks)
    if (len(peaks) == 0):
      print("\nClipping Detection No Peak ", fn)
      return 100
    clip_ratio = num_clipped/len(peaks)
    sorting_threshold = 0.3
    #return clip_ratio > threshold
    return clip_ratio

##################
#
##################
def join_cough_segments(audio: np.ndarray, fs: int, fn: str) -> np.ndarray:
  """Finds and joins cough segments.

  Args:
      audio (np.ndarray): audio
      fs (int): sample rate (should ideally be 44.1kHz)

  Returns:
      np.ndarray: a shorter audio clip just including coughs
  """
  filtered_audio = __filter_cough(audio, fs)
  decimated_fs = fs // 10
  resampled_audio, _ = resample_audio(filtered_audio, fs, )

  # Empiral mode decomposition => intrinsic mode functions
  imf = emd.sift.sift(resampled_audio, max_imfs=10)
  if (imf.shape[1] <= 4):
    print("cough segmentation failed ", fn)
    return audio, 1

  # Calculate and transform instantanious amplitudes
  _, _, IA_1 = emd.spectra.frequency_transform(imf[:, 1], decimated_fs, 'hilbert')
  _, _, IA_2 = emd.spectra.frequency_transform(imf[:, 4], decimated_fs, 'hilbert')  
  IA_1 = normalize_audio(IA_1)
  IA_2 = normalize_audio(IA_2)
  mean_IA = np.mean(np.hstack((IA_1, IA_2)), axis=-1)
  median_IA = signal.medfilt(mean_IA, kernel_size=51)

  # Normalize
  norm_IA = normalize_audio(median_IA)

  # Peak Detection & Segmentation
  peaks = signal.find_peaks(norm_IA, threshold=(-0.006), height=0.05)[0]
  mask, num_segments = __peaks2segments(norm_IA, peaks)
  
  # Upscale to original fs and segment original audio
  upscaled_mask = np.array([mask[i // 10] for i in range(len(audio))])
  mask_indices = np.argwhere(upscaled_mask == 1).flatten()
  #if len(mask_indices) == 0:
  #  print("cough not segmented ", fn)
  #  return audio, 1

  new_audio = audio[mask_indices]
  return new_audio, num_segments

##################
#
##################
def __peaks2segments(series: np.ndarray, peaks: list) -> np.ndarray:
  """Joins adjacent peaks to generate a segmentation mask.

  Args:
      series (np.ndarray): 1D series to create a mask for
      peaks (list): list of peaks in series

  Returns:
      np.ndarray: a boolean mask for segmentation
  """
  mask = np.zeros(series.shape[0])
  if len(peaks) <= 1:
      return mask, 1
  
  groups = []
  start = peaks[0]
  current_peak = peaks[0]
  for i, peak in enumerate(peaks[1:]):
      if peak - current_peak < 1500:
          current_peak = peak
      else:
          if peaks[i] != start:
              groups.append((start, peaks[i]))
          start = peak
          current_peak = peak
          
  groups.append((start, current_peak))
  for start, end in groups:
      if end - start > 400:
          mask[start-150:end+100] = 1
          
  return mask.astype(int), len(groups)

##################
#
##################
def __filter_cough(x: np.ndarray, fs: int) -> np.ndarray:
  """Applies signal filters to audio to crystallize cough frequencies.

  Args:
      x (np.ndarray): audio
      fs (int): sample rate

  Returns:
      np.ndarray: filtered audio
  """
  # 1st order 1kHz Butter highpass filter
  # goal: increase energy in low bands
  cutoff = 1000
  b, a = signal.butter(1, cutoff, 'lowpass', output='ba', fs=fs)
  filtered_x = signal.lfilter(b, a, x)

  # 2nd order 10Hz Chebyshev Type II highpass filter
  # goal: keep higher-pitch cough sounds while removing background noise
  cutoff = 10
  b, a = signal.cheby2(2, 40, cutoff, 'highpass', output='ba', fs=fs)
  filtered_x = signal.lfilter(b, a, filtered_x)

  return filtered_x


##################
#
##################
def resample_audio(audio: np.ndarray, orig_fs: int, target_fs=4410) -> Tuple:
  """Resamples the audio at the target sample rate.

  Args:
      audio (np.ndarray): audio
      orig_fs (int): original sample rate
      target_fs (int, optional): target sample rate. Defaults to 16000.


  Returns:
      Tuple: An audio-frequency tuple of the resampled audio
  """
  return librosa.resample(audio, orig_fs, target_fs, res_type='kaiser_best'), target_fs


##################
#
##################
def normalize_audio(x: np.ndarray, percentile: int=100) -> np.ndarray:
  """Normalizes audio volume.

  Args:
      x (np.ndarray): audio
      percentile (int): the percentile according to which the audio should be normalized

  Returns:
      np.ndarray: normalized audio
  """
  assert percentile >= 0 and percentile <= 100, 'The percentile must be between 0 and 100 inclusive.'
  if percentile == 100:
    return x / np.max(np.abs(x))

  norm_value = np.percentile(np.abs(x), percentile)
  return x / norm_value


##################
#
##################
def audio_feauture_extraction(audio, sr, row):  

  FV_LEN = 238
  
  S = np.abs(librosa.stft(audio))
  flatness = librosa.feature.spectral_flatness(S=S)
  contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=5) 
  chroma = librosa.feature.chroma_stft(S=S, sr=sr)
  duration = librosa.get_duration(y=audio, sr=sr)
  onset = librosa.onset.onset_detect(y=audio, sr=sr) 
  odf = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
  ac = librosa.autocorrelate(odf, max_size=1 * sr / 512)
  tempo = librosa.beat.tempo(y=audio,sr=sr)
  rmse = librosa.feature.rms(y=audio)
  rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
  spec_cent = librosa.feature.spectral_centroid(y=audio, S=S, sr=sr)
  zcr = librosa.feature.zero_crossing_rate(y=audio)
  mfccs = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=13)
  del_mfccs = librosa.feature.delta(mfccs, mode='nearest')
  #lpcv = librosa.lpc(audio, 3)
  #alphas = lpc_prep(torch.tensor(audio).unsqueeze(0))
  age = row['age']
  sex = 0 if row['biological_sex'] == 'male' else 1
  #label = 0 if row['pcr_test_result_inferred'] == 'negative' else 1


  fv = []    
  # first append label. It is a MUST for now. Not part of
  # feature vector. Better to return it separately but
  # couldn't do it with dask dataframe
  #fv.append(label)
  # second append duration. Also not a part of feature vector.
  #fv.append(duration)
    
  # now feature vectors  
  rmse_mean = rmse.mean(axis=1)
  rmse_min = rmse.min(axis=1)
  rmse_max = rmse.max(axis=1)
  assert(len(rmse_mean) == len(rmse_min))
  assert(len(rmse_min) == len(rmse_max))
  for i in range(len(rmse_mean)):
    fv.append(rmse_mean[i])
    fv.append(rmse_min[i])
    fv.append(rmse_max[i])   
 
  spectral_flatness_mean = flatness.mean(axis=1)
  spectral_flatness_min = flatness.min(axis=1)
  spectral_flatness_max = flatness.max(axis=1)
  assert(len(spectral_flatness_mean) == len(spectral_flatness_min))
  assert(len(spectral_flatness_min) == len(spectral_flatness_max))        
  for i in range(len(spectral_flatness_mean)):
    fv.append(spectral_flatness_mean[i])
    fv.append(spectral_flatness_min[i])
    fv.append(spectral_flatness_max[i])
  #print(spectral_flatness_mean.shape

  fv.append(tempo[0])
  # len(ac) depends on the sampling rate for now; need to keep it fixed
  ac_len = len(ac)
  ac_len_exp = 90
  if (ac_len > ac_len_exp):
    ac_len = ac_len_exp
  for i in range(ac_len):
    fv.append(ac[i])
  if (ac_len < ac_len_exp):
    for i in range(ac_len,ac_len_exp,1):
      fv.append(0)
  #print(ac.shape)

  # complains a lot about ill formed input
  #for i in range(len(lpcv)):
  #  fv.append(lpcv[i])
  #print(lpcv.shape)
  
  rolloff_mean = rolloff.mean(axis=1)
  rolloff_max = rolloff.max(axis=1)
  rolloff_min = rolloff.min(axis=1)
  assert(len(rolloff_mean) == len(rolloff_min))
  assert(len(rolloff_min) == len(rolloff_max))  
  #print(len(rolloff_mean))
  for i in range(len(rolloff_mean)):
    fv.append(rolloff_mean[i])
    fv.append(rolloff_min[i])
    fv.append(rolloff_max[i])
  #print(rolloff.shape)

  spec_cent_mean = spec_cent.mean(axis=1)
  spec_cent_max = spec_cent.max(axis=1)
  spec_cent_min = spec_cent.min(axis=1)
  assert(len(spec_cent_mean) == len(spec_cent_min))
  assert(len(spec_cent_min) == len(spec_cent_max))    
  #print(len(spec_cent_mean))
  for i in range(len(spec_cent_mean)):
    fv.append(spec_cent_mean[i])
    fv.append(spec_cent_min[i])
    fv.append(spec_cent_max[i])
  #print(spec_cent.shape)
   
  zcr_mean = zcr.mean(axis=1)
  zcr_min = zcr.min(axis=1)
  zcr_max = zcr.max(axis=1)   
  assert(len(zcr_mean) == len(zcr_min))
  assert(len(zcr_min) == len(zcr_max))     
  #print(len(zcr_mean))   
  for i in range(len(zcr_mean)):
    fv.append(zcr_mean[i])
    fv.append(zcr_min[i])
    fv.append(zcr_max[i])
    
  mfccs_mean = mfccs.mean(axis=1)
  mfccs_min = mfccs.min(axis=1)
  mfccs_max = mfccs.max(axis=1)  
  assert(len(mfccs_mean) == len(mfccs_min))
  assert(len(mfccs_min) == len(mfccs_max))       
  for i in range(len(mfccs_mean)):
    fv.append(mfccs_mean[i])
    fv.append(mfccs_min[i])
    fv.append(mfccs_max[i])  
  #print(mfccs_mean.shape)
  
  del_mfccs_mean = del_mfccs.mean(axis=1)
  del_mfccs_min = del_mfccs.min(axis=1)
  del_mfccs_max = del_mfccs.max(axis=1)
  assert(len(del_mfccs_mean) == len(del_mfccs_min))
  assert(len(del_mfccs_min) == len(del_mfccs_max))         
  for i in range(len(del_mfccs_mean)):
    fv.append(del_mfccs_mean[i])
    fv.append(del_mfccs_min[i])
    fv.append(del_mfccs_max[i])
  #print(del_mfccs_mean.shape)

  contrast_mean = contrast.mean(axis=1)
  contrast_min = contrast.min(axis=1)
  contrast_max = contrast.max(axis=1)
  assert(len(contrast_mean) == len(contrast_min))
  assert(len(contrast_min) == len(contrast_max))    
  for i in range(len(contrast_mean)):
    fv.append(contrast_mean[i])
    fv.append(contrast_min[i])
    fv.append(contrast_max[i])   
  #print(contrast_mean.shape)

  chroma_mean = chroma.mean(axis=1)
  chroma_min = chroma.min(axis=1)
  chroma_max = chroma.max(axis=1)
  assert(len(chroma_mean) == len(chroma_min))
  assert(len(chroma_min) == len(chroma_max))    
  for i in range(len(chroma_mean)):
    fv.append(chroma_mean[i])
    fv.append(chroma_min[i])
    fv.append(chroma_max[i])    
  #print(chroma_mean.shape) 
    
  if (len(fv) != FV_LEN):
    exit("invalid FV_LEN")
    
  return np.array(fv)

           
##################
#
##################
def load_and_check_audio(row):

  sound_file_path = row['audio_path']  

  if not os.path.exists(sound_file_path):
    print("\nmissing file ", sound_file_path)
    return None, None, -1

  # use the original sampling rate of the audio file
  audio, sr = librosa.load(sound_file_path, sr = None)

  #if len(audio) < 2048:
  #  print("\nshort audio signal ", sound_file_path)
  #  return None, None, -1

  #clipping_detection = clipping_detect(audio, sr, sound_file_path, 0.3)
  #if (clipping_detection > 0.5):
  #  print("\nclipping detected ", sound_file_path)
  #  return None, None, -1

  resampled_audio = librosa.resample(audio, sr, 44100)
  just_cough_audio, num_segments = join_cough_segments(resampled_audio, 44100, sound_file_path)
  if len(just_cough_audio) < 2048:
    just_cough_audio = audio
  else:
    sr = 44100

  return just_cough_audio, sr, 0
 

##################
#
##################
def row_apply_load_check_predict_audio(row, clf):

  just_cough_audio, sr, err = load_and_check_audio(row)
  if err == -1:
    return dict({'fv': np.zeros(238), 'audio_path': row['audio_path']})

  fv = audio_feauture_extraction(just_cough_audio, sr, row)
  #return dict({'pcr_test_result_predicted': fv, 'audio_path': row['audio_path']})
  return dict({"fv":np.array(fv), "audio_path":row['audio_path']})

##################
#
##################
def df_apply_load_check_predict_audio(df, clf):

  return df.apply(lambda row: row_apply_load_check_predict_audio(row, clf), axis=1)



##################
#
##################
def SVM_Predict(model_dir:str, df:pd.DataFrame):
    
  
  # load the model
  clf = load(model_dir)


  result = subprocess.check_output('nproc --all', shell = True)
  num_proc = int(result)
  print("number of processors:", num_proc)

  # create dask dataframe for parallel processing
  ddata = dd.from_pandas(df, npartitions=num_proc)
  meta_types = df.dtypes
  meta = daskutils.make_meta(meta_types, index=pd.Index([], 'i8'))
  # partition the original datafram
  dmap = ddata.map_partitions(lambda df: df_apply_load_check_predict_audio(df, clf), meta=meta)
  with ProgressBar():
    response = dmap.compute(scheduler=get)
  response = response.to_list()
  response_df = pd.DataFrame(data=response, columns={'fv', 'audio_path'})
  #response_df = pd.DataFrame.from_records(response, columns =['fv', 'audio_path'])
  #print(np.vstack(response_df['fv'].to_numpy()).shape)
  #print(type(np.vstack(response_df['fv'].to_numpy()))) 
  #print(type(response_df.iloc[:,0]))

  response_df['pcr_test_result_predicted'] = clf.predict(np.vstack(response_df['fv'].to_numpy()))
  return response_df

    


####################
# TEST THE PREDICTION API. NOT SPLITTED AS VALIDATION AND TRATN THOUGH
####################

if __name__ == '__main__':
  freeze_support()

  data_dir = "/mnt/disks/nvme3n1/data/"
  output_dir = "/home/desin/audio-models/out/" 

  dataset_names = ( 
    'virufy-cdf-iatos',
    'virufy-cdf-coswara',
    'virufy-cdf-coughvid',    
    'virufy-cdf-crowdsource',
  )

  datasets = []
  for dataset_name in dataset_names:
    df = pd.read_csv(data_dir + "/" + dataset_name + "-extras.csv")
    #if dataset_name == 'virufy-cdf-crowdsource':
    #   df['source'] =  df.apply(lambda row: return 'crowdsource', axis=1)
    # remove unlabeled ones
    df = df[(df['pcr_test_result_inferred'] == 'negative') | (df['pcr_test_result_inferred'] == 'positive')]
    df['audio_path'] = df['audio_path'].apply(lambda x: data_dir + "/" + dataset_name + "/" + x)
    df['pcr_test_result_inferred'] = df['pcr_test_result_inferred'].apply(lambda x: 1 if x == 'positive' else 0)  
    datasets.append(df)

  df = pd.concat(datasets)
  df_all = df


  #print(f"\nprocessing {dataset_name}")
  #df = pd.read_csv(data_dir + "/" + dataset_name + "/" + dataset_name + "-label-extras.csv")
  print("original size:{}".format(len(df)))

    
  val_prediction_df = SVM_Predict('svm.joblib', df.drop([
    'pcr_test_result',
    'pcr_test_result_inferred'], axis=1))
  
  labels_and_predictions_df = val_prediction_df.merge(
          right = df_all,
    on = 'audio_path',
    how = 'inner'
  )
  labels_and_predictions_df.info()


  print(sklearn.metrics.classification_report(labels_and_predictions_df['pcr_test_result_inferred'],labels_and_predictions_df['pcr_test_result_predicted']))
