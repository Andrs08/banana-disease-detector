import os
import hashlib
import torch
import torch.nn as nn
from src.training.dataset import get_data_loaders 
from src.training.evaluate import evaluate
from src.training.model import get_model

DATASET_DIR = "data/dataset"


def get_image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_hashes(split):
    hashes = set()
    split_path = os.path.join(DATASET_DIR, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            hashes.add(get_image_hash(img_path))

    return hashes


def check_leakage():
    train_hashes = get_hashes("train")
    val_hashes = get_hashes("val")
    test_hashes = get_hashes("test")

    print("Comparando datasets...\n")

    train_val = train_hashes.intersection(val_hashes)
    train_test = train_hashes.intersection(test_hashes)
    val_test = val_hashes.intersection(test_hashes)

    print(f"Train vs Val: {len(train_val)} duplicados")
    print(f"Train vs Test: {len(train_test)} duplicados")
    print(f"Val vs Test: {len(val_test)} duplicados")

    test_loader, _, classes = get_data_loaders(
        "data/dataset/test",
        "data/dataset/test"  
    )

    model = get_model(num_classes=len(classes))
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    # función de pérdida
    criterion = nn.CrossEntropyLoss()

    # evaluar
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print("="*30)
    print("RESULTADOS EN TEST")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*30)


if __name__ == "__main__":
    check_leakage()