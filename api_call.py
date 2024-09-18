import requests
import json
import pandas as pd

# 모델 예측을 위한 입력 데이터 준비
input_songs = ['뉴진스 - Attention']  # 실제 곡 제목으로 예측 데이터 준비
input_data = pd.DataFrame(input_songs, columns=["input_song"])

# DataFrame을 JSON 형식으로 변환
json_data = input_data.to_json(orient="split")

# API 호출
api_url = "http://13.125.252.79:1234/invocations"
headers = {"Content-Type": "application/json"}
response = requests.post(api_url, headers=headers, data=json_data)

# 결과 확인
if response.status_code == 200:
    predictions = response.json()
    print("Predicted next songs:", predictions)
else:
    print(f"Error: {response.status_code}, {response.text}")
