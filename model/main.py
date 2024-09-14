import json
with open("/home/ubuntu/anonimous_video_comments.json",'r') as f:
    playlists = json.load(f)

#플레이리스트 하나를 하나의 문장으로 취급
# 문장에 들어가야 하는 값은 ['단어','단어','단어']형태로 들어있어야 함

from src.Preprocessing import preprocessing
from src.Dataset import LSTMdataset
from src.LSTMModel import LSTMModel

playlists = preprocessing(playlists)

dataset = LSTMdataset(playlists = playlists)

model = LSTMModel(num_embeddings=len(dataset.BOM))            


from torch.utils.data.dataloader import DataLoader
loader = DataLoader(dataset, batch_size=16)
import torch.nn as nn
import torch
# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 학습
from src.Train import train_model
train_model(model, loader, criterion, optimizer)

from sklearn.preprocessing import LabelEncoder

all_songs = [song for playlist in playlists for song in playlist]

label_encoder = LabelEncoder()
label_encoder.fit(all_songs)


from src.Predict import predict_next_songs, predict_one_song
# 7. 다음 노래 예측 함수
input_text = "뉴진스 - Attention"
predicted_song = predict_one_song(model, input_text, label_encoder)
predicted_songs = predict_next_songs(model, input_text, label_encoder=label_encoder, top_k=10)

print(f'One recommended song:{predicted_song}')
print(f"Next recommended song: {predicted_songs}")


