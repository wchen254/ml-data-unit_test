
ssh -f -N -L 8898:localhost:8888 gateway.awxyz.net -p 2249

sudo docker run --gpus all --ipc=host -it --rm -p 8888:8888 -v /home/desin/audio-models:/tf/audio-models tf-2.4-gpu-jupyter-s3prl bash

sudo docker run --gpus device=1 --ipc=host -it --rm -p 8888:8888 -v /home/desin/audio-models:/tf/audio-models    tf-2.4-gpu-jupyter-s3prl bash
 sudo docker exec -it  dazzling_keller bash

pip install requirements.txt
apt-get install libsndfile1

python3 preprocess/generate_len_for_bucket.py --audio_extension .wav -i /tf/audio-models/trimmed_dataset


python run_pretrain.py -u tera -g pretrain/tera/config_model.yaml -n covidModel-2 --multi_gpu

python run_pretrain.py -u tera -g pretrain/tera/exp_model_config/fbankBase-T-F.yaml -n covidModel-2 --multi_gpu


python run_pretrain.py -e /tf/audio-models/s3prl/result/pretrain/covidModel-3

python3 run_downstream.py -n exp-2 -m train -u tera_local -s 2  -k /tf/audio-models/s3prl/result/pretrain/covidModel-2/states-epoch-229.ckpt -d cough_classify -o "config.runner.eval_dataloaders=['train', 'test']"


python3 utility/get_best_dev.py result/downstream/exp-1/log.log
