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

# artifact_store_path = "file:///home/ubuntu/feature_store"
# mlflow.set_tracking_uri(artifact_store_path)



#전처리
from src.Preprocessing import preprocessing
from src.Dataset import LSTMdataset

playlists = preprocessing(playlists)
dataset = LSTMdataset(playlists = playlists)
print()

#모델 학습
from src.Train import run_training

run_training(dataset=dataset)


#학습된 모델 정보 불러오기

# for version in versions:
#     print(f"Model Name: {version.name}")
#     model_name = version.name
#     print(f"Version: {version.version}")
#     model_version = version.version
#     print(f"Stage: {version.current_stage}")
#     print(f"Run ID: {version.run_id}")
#     print(f"Status: {version.status}")


# #모델 서빙
# import mlflow.pyfunc
# import requests
# import pandas as pd



# model_uri = f"models:/{model_name}/{model_version}"
# model = mlflow.pyfunc.load_model(model_uri)
# print(model)

# # Input data - here we assume that you are predicting based on song names
# input_songs = ['뉴진스 - Attention'] # Replace with actual songs

# # Prepare the data in the required format (as a pandas DataFrame)
# input_data = pd.DataFrame(input_songs, columns=["input_song"])

# # Convert the input data to JSON format, which is required by the MLflow model serving endpoint
# json_data = input_data.to_json(orient="split")

# # Send the request to the MLflow serving endpoint
# response = requests.post(model_uri, headers={"Content-Type": "application/json"}, data=json_data)

# # # Check the response
# if response.status_code == 200:
#     # Print the predictions
#     predictions = response.json()
#     print("Predicted next songs:", predictions)
# else:
#     # Print error message
#     print(f"Error: {response.status_code}, {response.text}")


# # from sklearn.preprocessing import LabelEncoder

# # all_songs = [song for playlist in playlists for song in playlist]

# # label_encoder = LabelEncoder()
# # label_encoder.fit(all_songs)


# # from src.Predict import predict_next_songs, predict_one_song
# # # 7. 다음 노래 예측 함수
# # input_text = "뉴진스 - Attention"
# # predicted_song = predict_one_song(model, input_text, label_encoder)
# # predicted_songs = predict_next_songs(model, input_text, label_encoder=label_encoder, top_k=10)

# # print(f'One recommended song:{predicted_song}')
# # print(f"Next recommended song: {predicted_songs}")
