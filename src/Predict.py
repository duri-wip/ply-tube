# import torch

# def predict_next_songs(model, input_song, label_encoder,top_k=10):
#     model.eval()
#     input_seq = torch.tensor(label_encoder.transform([input_song]), dtype=torch.long)
#     with torch.no_grad():
#         output = model(input_seq)
#         predicted_index = torch.topk(output, top_k, dim=1).indices.squeeze().tolist()
#     predicted_song = label_encoder.inverse_transform(predicted_index)

#     return predicted_song

# def predict_one_song(model, input_song, label_encoder):
#     model.eval()
#     input_seq = torch.tensor(label_encoder.transform([input_song]), dtype=torch.long)
#     with torch.no_grad():
#         output = model(input_seq)
#         predicted_index = torch.argmax(output, dim=1).item()
#     return label_encoder.inverse_transform([predicted_index])[0]

import mlflow.pyfunc
import torch

class SongPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, input_df):
        # Assumes input_df is a DataFrame with a single column 'input_song'
        input_song = input_df['input_song'].tolist()
        predictions = [self.predict_next_songs(song) for song in input_song]
        return predictions
    
    def predict_next_songs(self, input_song, top_k=10):
        self.model.eval()
        input_seq = torch.tensor(self.label_encoder.transform([input_song]), dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_seq)
            predicted_index = torch.topk(output, top_k, dim=1).indices.squeeze().tolist()
        predicted_song = self.label_encoder.inverse_transform(predicted_index)
        return predicted_song


모델
학습 - 테스트 : 모델 저장
예측 

curl -X POST -H "Content-Type:application/json; format=pandas-split" \
--data '{"columns":["column1", "column2"], "data":[[value1, value2]]}' \
http://12.34.56.78:1234/invocations