#데이터 로드 - 샘플
import json
with open("/home/ubuntu/anonimous_video_comments.json",'r') as f:
    playlists = json.load(f)


#mlflow tracking uri 설정
from dotenv import load_dotenv
import os
import mlflow

load_dotenv(verbose=True, override=True)
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
print(os.getenv('MLFLOW_TRACKING_URI'))

#전처리
from src.Preprocessing import preprocessing
from src.Dataset import LSTMdataset

playlists = preprocessing(playlists)
dataset = LSTMdataset(playlists = playlists)
print()

#모델 학습
from src.Train import run_training

run_training(dataset=dataset)