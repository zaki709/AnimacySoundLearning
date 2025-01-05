import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_preprocessing.dataloader import AnimacyDataset
from models.resnet_model import initialize_model
from models.train import train_model
from notification.notify import send_email

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = AnimacyDataset(
        csv_file_path="data/metadata/dataset_metadata.csv", transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = initialize_model(model_name="resnet34", num_classes=2)
    train_model(
        model,
        dataloader,
        num_epochs=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    send_email("train is done")
