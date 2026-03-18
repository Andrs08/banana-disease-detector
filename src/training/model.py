import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # reemplazar última capa del modelo
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model