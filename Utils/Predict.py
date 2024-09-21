import torch

# 예측을 수행하는 함수 정의
def predict_songs(label_encoder, model, device, input_song, top_k=10):
    input_seq = torch.tensor(label_encoder.transform([input_song]), dtype=torch.long)
    input_seq = input_seq.unsqueeze(0).to(device)  # 배치 차원 추가 및 장치 이동

    with torch.no_grad():
        output = model(input_seq)
        predicted_indices = torch.topk(output, top_k*2, dim=1).indices.squeeze().tolist()

    # 유효한 예측 인덱스 선택
    valid_predicted_indices = []
    for idx in predicted_indices:
        if idx < len(label_encoder.classes_):  # 유효한 인덱스인지 확인
            valid_predicted_indices.append(idx)
        if len(valid_predicted_indices) == top_k:  # 유효한 값이 10개가 되면 중단
            break

    # 예측된 인덱스를 실제 노래로 변환
    if valid_predicted_indices:
        predicted_songs = label_encoder.inverse_transform(valid_predicted_indices)
        predicted_songs = predicted_songs.tolist()  # 리스트로 변환
        return predicted_songs
    else:
        return []