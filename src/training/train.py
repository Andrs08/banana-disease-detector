import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import get_model
from evaluate import evaluate

def train():

    train_loader, val_loader, classes = get_data_loaders(
        "data/dataset/train",
        "data/dataset/val"
    )
    model = get_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    best_acc = 0
    epochs = 10
    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        # guardar el mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/model.pth")

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {running_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print("-"*30)

if __name__ == "__main__":
    train()