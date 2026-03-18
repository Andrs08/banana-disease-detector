import torch

def evaluate(model, dataloader, criterion):

    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():

        for images, labels in dataloader:

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    
    return running_loss, accuracy