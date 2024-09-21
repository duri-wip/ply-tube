import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException


from sklearn.metrics import precision_score, recall_score, f1_score

from src.Dataset import build_dataloaders
from src.LSTMModel import LSTMModel

def train_model(model, dataset, dataloader, criterion, optimizer):
    size = len(dataset)
    model.train()
    for batch, (X_batch, Y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(X_batch)

        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X_batch)
            print(f"Loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

def test_model(model, dataset, dataloader, loss_fn):
    model.eval()
    total_loss, total_correct = 0, 0
    total_samples = len(dataset)  # 전체 데이터셋 크기
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            # 모델 예측
            pred = model(X_batch)
            
            # 배치 손실 계산 (배치 크기만큼 곱해준 후 나중에 전체 평균)
            batch_loss = loss_fn(pred, Y_batch).item() * X_batch.size(0)
            total_loss += batch_loss  # 총 손실에 더함
            
            # 배치 정확도 계산
            correct_predictions = (pred.argmax(dim=1) == Y_batch).sum().item()
            total_correct += correct_predictions  # 맞춘 예측 수 더하기
            
            # 예측값과 실제값 저장
            all_preds.extend(pred.argmax(dim=1).cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    # 전체 평균 손실 계산 (전체 샘플 수로 나누기)
    avg_loss = total_loss / total_samples

    # 전체 정확도 계산 (정확하게 예측한 샘플 수를 전체 샘플 수로 나눔)
    accuracy = total_correct / total_samples * 100

    # Precision, Recall, F1 Score 계산
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    print(f"Test Results: \n Accuracy: {accuracy:.2f}%, Avg loss: {avg_loss:.6f}")
    print(f"Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1:.6f}\n")

    return accuracy, avg_loss, precision, recall, f1

def run_training(dataset, epochs=10, learning_rate=1e-3, batch_size=64):    
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(dataset, batch_size=batch_size)
    
    model = LSTMModel(num_embeddings=len(train_dataset.dataset.BOM))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)

    best_acc = 0.0

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n--------------------------")
            train_model(model, train_dataset, train_loader, loss_fn, optimizer)
            test_acc, test_loss, precision, recall, f1 = test_model(model, test_dataset, test_loader, loss_fn)

            # MLflow에 지표 로그
            mlflow.log_metric('Accuracy', test_acc, step=epoch)
            mlflow.log_metric('Loss', test_loss, step=epoch)
            mlflow.log_metric('Precision', precision, step=epoch)
            mlflow.log_metric('Recall', recall, step=epoch)
            mlflow.log_metric('F1Score', f1, step=epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                print(f'new best accuracy : {best_acc}')
                mlflow.log_metric('best_acc', best_acc, step=epoch)
                input_tensor = torch.tensor(train_dataset[0][0]).unsqueeze(0).detach().cpu().numpy()
                model_output = model(torch.tensor(train_dataset[0][0]).unsqueeze(0)).detach().cpu().numpy()

                signature = infer_signature(input_tensor, model_output)
                mlflow.pytorch.log_model(model.cpu(), 'model', signature=signature)

            scheduler.step()

        input_tensor = torch.tensor(train_dataset[0][0]).unsqueeze(0).detach().cpu().numpy()
        model_output = model(torch.tensor(train_dataset[0][0]).unsqueeze(0)).detach().cpu().numpy()

        signature = infer_signature(input_tensor, model_output)
        mlflow.pytorch.log_model(model.cpu(), "model", signature=signature)

        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Model successfully logged at {model_uri}")

        model_name = "LSTMModel"
        try:
            mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"Model registered as '{model_name}' in model registry.")
        except MlflowException as e:
            print(f"Model registration failed: {e}")

        print("Done!")
        mlflow.end_run()
