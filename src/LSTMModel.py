import torch.nn as nn
import torch
from mlflow.models.signature import infer_signature


class LSTMModel(nn.Module):
    def __init__(self, num_embeddings):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=16)
        self.lstm = nn.LSTM(
                input_size=16,
                hidden_size=64,
                num_layers=5,
                batch_first=True)
        
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_embeddings)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.lstm(x)

        x = torch.reshape(x, (x.shape[0],-1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x