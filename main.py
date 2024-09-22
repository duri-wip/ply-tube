#데이터 로드
import json
with open("/home/ubuntu/feature_store/playlist.json",'r') as f:
    playlists = json.load(f)
print("-----------데이터 로드-----------")

#mlflow tracking uri 설정
from dotenv import load_dotenv
import os
import mlflow

load_dotenv(verbose=True, override=True)
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
print("-----------tracking uri 설정-----------")
print(os.getenv('MLFLOW_TRACKING_URI'))

# #전처리
# from src.Preprocessing import preprocessing
# from src.Dataset import LSTMdataset

# playlists = preprocessing(playlists)

# dataset = LSTMdataset(playlists = playlists)


# #모델 학습
# from src.Train import run_training
# run_training(dataset=dataset)

# 운영중인 모델과 비교해서 새로운 모델로 변경
from utils.ProductionAlias import production_alias
model_name = "LSTMModel"

production_alias(model_name=model_name, param='Loss')