import os
import torch
import mlflow
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from utils.ProductionModelLoad import load_production_model_by_stage
from utils.AllSongs import generate_all_songs
from utils.Predict import predict_songs


# Flask 애플리케이션 초기화
app = Flask(__name__)

# 환경 변수 로드 및 MLflow 설정
load_dotenv(verbose=True, override=True)
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)

# 운영중인 모델 찾기
model_name = 'LSTMModel'
model_uri = load_production_model_by_stage(model_name)

# 모델 로드
model = mlflow.pytorch.load_model(model_uri)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 평가모드로 전환
model.eval()

# LabelEncoder 로드
label_encoder = generate_all_songs()

# API 엔드포인트 정의
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_song = data.get('input_song', None)
        
        if not input_song:
            return jsonify({"error": "No input song provided"}), 400
        
        # 노래 예측 수행
        predicted_songs = predict_songs(label_encoder, model, device, input_song)
        return jsonify({"predicted_songs": predicted_songs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # 모델과 장치가 정상적으로 로드되었는지 확인
        if model and label_encoder and device:
            return jsonify({"status": "healthy"}), 200
        else:
            return jsonify({"status": "unhealthy"}), 500
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
