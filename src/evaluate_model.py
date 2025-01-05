import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from data_preprocessing.dataloader import AnimacyDataset
from models.resnet_model import initialize_model


def load_model(model_path, model_name="resnet34", num_classes=2, device="cpu"):
    """
    Load the saved model from the checkpoint.

    Args:
        model_path (str): Path to the saved model checkpoint.
        model_name (str): Model architecture name (e.g., "resnet34").
        num_classes (int): Number of output classes.
        device (str): Device to load the model on.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model = initialize_model(model_name=model_name, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate the model.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        device (str): Device to evaluate on.

    Returns:
        None
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    # Configuration
    model_path = "output/models/best_model.pth"  # Path to the best saved model
    model_name = "resnet34"  # ResNet variant used during training
    num_classes = 2  # Number of output classes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DataLoader for evaluation
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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the model
    model = load_model(
        model_path=model_path,
        model_name=model_name,
        num_classes=num_classes,
        device=device,
    )

    # Evaluate the model
    evaluate_model(model, dataloader, device=device)
