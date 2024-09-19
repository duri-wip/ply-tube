import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import mlflow

load_dotenv(verbose=True, override=True)
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
print(os.getenv('MLFLOW_TRACKING_URI'))

client = MlflowClient()

model_name = "LSTMModel"
versions = client.get_registered_model(model_name).latest_versions

model_name = [version.name for version in versions]
model_uri = f"models:/{model_name[0]}/{13}"

print(model_uri)


model = mlflow.pyfunc.load_model(model_uri)
print(model)

# # 예측할 입력 데이터 준비
# import pandas as pd
# import torch

# from sklearn.preprocessing import LabelEncoder

# import json
# with open("/home/ubuntu/anonimous_video_comments.json",'r') as f:
#     playlists = json.load(f)

# from src.Preprocessing import preprocessing
# playlists = preprocessing(playlists)
# all_songs = [song for playlist in playlists for song in playlist]

# label_encoder = LabelEncoder()
# label_encoder = label_encoder.fit(all_songs)

# import numpy as np
# input_song = ['Lauv - Sims']  # 실제 곡 제목으로 예측 데이터 준비
# input_seq = label_encoder.transform(input_song)  # 인덱스 번호로 변환

# # 변환된 인덱스 번호를 numpy 배열로 변환
# input_array = np.array(input_seq).reshape(-1, 1)  # 모델이 2D 배열을 예상할 수 있으므로 reshape
# #print(input_array)
# # numpy 배열을 pandas DataFrame으로 변환
# input_df = pd.DataFrame(input_array, columns=["song_index"])
# #print(input_df)
# # 모델을 사용하여 예측
# predictions = model.predict(input_df)

# predictions = np.array(predictions)
# top_k = 10
# top_k_indices = np.argsort(predictions[0])[-top_k:]

# # 상위 10개 인덱스의 확률 값 추출
# top_k_probabilities = predictions[0][top_k_indices]

# # 상위 10개 인덱스를 원래 레이블로 변환
# top_k_songs = label_encoder.inverse_transform(top_k_indices)

# # 상위 10개 예측 결과와 확률 출력
# for song, prob in zip(top_k_songs, top_k_probabilities):
#     print(f"Song: {song}, Probability: {prob}")
# # print("Raw predictions:", predictions)

# # # 예측 결과가 pandas DataFrame일 경우, numpy 배열로 변환
# # if isinstance(predictions, pd.DataFrame):
# #     predictions = predictions.values

# # # 예측 결과가 numpy 배열일 경우, 1D 배열로 변환
# # if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
# #     predictions = predictions.flatten()

# # predictions = predictions.astype(int)

# # # label_encoder.inverse_transform 적용
# # predicted_song = label_encoder.inverse_transform(predictions)

# # # 예측 결과 출력
# # print("Predicted next songs:", predicted_song)
# # # 예측 결과 출력
# # print("Flattened predictions:", predictions)

# # # label_encoder.inverse_transform 적용
# # predicted_song = label_encoder.inverse_transform(predictions.astype(int))

# # # 예측 결과 출력
# # print("Predicted next songs:", predicted_song)
# # def predict_next_songs(model, input_song, label_encoder,top_k=10):
# # #     model.eval()
# # #     with torch.no_grad():
# # #         output = model(input_seq)
# # #         predicted_index = torch.topk(output, top_k, dim=1).indices.squeeze().tolist()
# # #     predicted_song = label_encoder.inverse_transform(predicted_index)
