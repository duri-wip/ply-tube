import json
with open("/home/ubuntu/anonimous_video_comments.json",'r') as f:
    playlists = json.load(f)

from src.Preprocessing import preprocessing
from src.Dataset import LSTMdataset
from src.LSTMModel import LSTMModel

playlists = preprocessing(playlists)
dataset = LSTMdataset(playlists = playlists)
from torch.utils.data import DataLoader, random_split

def build_dataloaders(dataset, batch_size = 64):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


train_loader, test_loader = build_dataloaders(dataset=dataset, batch_size=64)

model = LSTMModel(num_embeddings=len(dataset.BOM))

import torch.nn as nn
import torch

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 학습
import sys
from src.Train import run_training

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
BOM = dataset.BOM
run_training(BOM, train_loader=train_loader, test_loader=test_loader, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)


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

