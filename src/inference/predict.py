import torch
from torchvision import transforms
from PIL import Image

from src.training.model import get_model

classes = ["healthy", "sigatoka"]

model = get_model(num_classes=len(classes))
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

def preprocess_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension

    return image


def predict(image_path):

    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    confidence = probabilities[0][predicted.item()].item()

    return {
        'prediction': classes[predicted.item()],
        'probabilidad': confidence
    }
    
