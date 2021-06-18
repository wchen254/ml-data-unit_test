
Notes:

1. There is no model directory as an input to the predict api. Model directory is hardcoded in the code right now.

2. Extract the s3prl directory and build docker image  inside s3prl

	cd s3prl
	docker build . -t virufy_s3prl

3. Run the docker image virufy_s3prl with an input of the .csv file location and name. You need to map the local 
directory for .csv to the docker directory (/home/virufy_predict_api/data). Directory for the .flac audio files
should be under the same directory with the .csv file so that docker image can also access audio files

	docker run --gpus all  -v LOCAL_PATH_TO_CSV_AND_DATA_DIR:/home/virufy_predict_api/data  virufy_s3prl /home/virufy_predict_api/data/virufy-cdf-coughvid-val.csv

4. You can also enter to the shell of docker container and then run the code as below

	docker run -ti --gpus all -v LOCAL_PATH_TO_CSV_AND_DATA_DIR:/home/virufy_predict_api/data  virufy_s3prl bash
	python3 run_predict.py /home/virufy_predict_api/data/virufy-cdf-coughvid-val.csv

5. If there is no gpu in the system cpu will be used. Loading dictioanary files (model files in pytorch) and especially 
position encoding table for the first time takes a long time, ~ 1 minute. 

6. This version is trained on some of the coughvid data and tested on it. Results are not good. 
There might be ways to improve though.
