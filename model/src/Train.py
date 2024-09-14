def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            #print(f"Outputs shape: {output.shape}, Y_batch shape: {Y_batch.shape}")  # 디버깅용 출력

            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
