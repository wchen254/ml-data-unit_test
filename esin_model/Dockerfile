FROM tensorflow/tensorflow:2.4.1

## Source code (API, model)
WORKDIR /home/virufy_predict_api
COPY LICENSE .
COPY README.md .
ADD downstream /home/virufy_predict_api/downstream
COPY hubconf.py .
COPY optimizers.py .
COPY preprocess /home/virufy_predict_api/preprocess
COPY requirements.txt .
COPY result /home/virufy_predict_api/result
COPY run_predict.py .
COPY schedulers.py .
COPY upstream /home/virufy_predict_api/upstream
COPY utility /home/virufy_predict_api/utility

RUN pip install --user -r requirements.txt
RUN apt-get update
RUN apt-get install -y libsndfile1 libgl1-mesa-glx

ENV PATH=/root/.local:$PATH


EXPOSE 5000

ENTRYPOINT ["python3", "run_predict.py"]

