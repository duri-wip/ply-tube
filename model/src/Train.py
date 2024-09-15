import os
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

import mlflow
import mlflow.pytorch

from src.LSTMModel import build_model
from src.Dataset import build_dataloaders

load_dotenv(verbose=True, override=True)
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

#tracking_uri = os.environ.get('MLFLOW_TRACKING_URI','')
mlflow.set_tracking_uri(tracking_uri)

def train_model(model, dataloader, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X_batch, Y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(X_batch)
        #print(f"Outputs shape: {output.shape}, Y_batch shape: {Y_batch.shape}")  # 디버깅용 출력

        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_model(model, dataloader, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            pred = model(X_batch)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
    
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    test_acc = correct * 100.0

    print(f"Test error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_acc, test_loss

def run_training(epochs=10, learning_rate=1e-2, batch_size=64):
    train_loader, test_loader = build_dataloaders(batch_size)
    model, signature = build_model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    with mlflow.start_run():
        mlflow.log_param("epochs",epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n--------------------------")
            train_model(train_loader, model, loss_fn, optimizer)
            test_acc, test_loss = test_model(test_loader, model, loss_fn)

            mlflow.log_metric("test_acc", test_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                mlflow.log_metric("best_acc", best_acc, step=epoch)
                mlflow.pytorch.log_model(model.cpu(), "model", signature=signature, code_paths=['src/LSTMModel.py'])
                
    print("Done!")