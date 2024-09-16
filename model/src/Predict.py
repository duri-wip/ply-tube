import torch

def predict_next_songs(model, input_song, label_encoder,top_k=10):
    model.eval()
    input_seq = torch.tensor(label_encoder.transform([input_song]), dtype=torch.long)
    with torch.no_grad():
        output = model(input_seq)
        predicted_index = torch.topk(output, top_k, dim=1).indices.squeeze().tolist()
    predicted_song = label_encoder.inverse_transform(predicted_index)
    return predicted_song

def predict_one_song(model, input_song, label_encoder):
    model.eval()
    input_seq = torch.tensor(label_encoder.transform([input_song]), dtype=torch.long)
    with torch.no_grad():
        output = model(input_seq)
        predicted_index = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_index])[0]


