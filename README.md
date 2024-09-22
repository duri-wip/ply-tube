# Ply-tube

## About

노래, 가수를 고르면 새로운 플레이리스트를 추천해드립니다!

사용자 정보, 노래의 메타 데이터 없이 느낌있는 플레이리스트 추천하기

## Goals

- Youtube API를 이용하여 '플레이리스트' 콘텐츠를 제작하는 채널의 정보 수집
- 위 채널들에서 올린 영상 정보를 수집하여 플레이리스트 제목과 그 안의 노래 정보 수집
- LSTM 모델을 활용하여 플레이리스트 추천 모델링
- 랜덤하게 노래 제목과 가수 목록을 제공하고, 사용자가 선택한 정보를 바탕으로 관련 있는 플레이리스트를 추론하여 제공

## Architecture

![아키텍처 구조도]()

## Environment

### ![Pipeline server] (링크..)

| Tool/Technology | Description/Purpose                       | Version |
| --------------- | ----------------------------------------- | ------- |
| Airflow         | 데이터 수집 및 파이프라인 워크플로우 관리 |         |
| Postgres        | RDB, 1차 전처리된 raw데이터 저장          |         |
| Pgpool          | PostgreSQL 이중화 구현                    |         |
| DVC             | Data version control                      |         |
| FastAPI         | 백엔드 서버 Web framework                 |         |

### ![Model server] (링크)

| Tool/Technology | Description/Purpose     | Version |
| --------------- | ----------------------- | ------- |
| DVC             | Data version control    |         |
| MLflow          | 머신러닝 lifecycle 관리 |         |
| Pyenv           | Python 가상환경관리     |         |
| Python          | pytorch, mlflow를 수행  |         |
| GitHub Actions  | CI/CD automation        |         |
