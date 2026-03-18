from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_folder, val_folder, batch_size=32):

    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        train_folder,
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        val_folder,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader, train_dataset.classes