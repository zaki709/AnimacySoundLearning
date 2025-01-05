import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def save_checkpoint(model, epoch, path="checkpoint.pth"):
    """Save the model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def train_model(model, dataloader, num_epochs=10, lr=0.001, patience=3, device=""):
    """
    Train the model with early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        dataloader (DataLoader): DataLoader for training data.
        num_epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        patience (int): Number of epochs to wait for improvement before stopping.
        device (str): Device to train on ("cpu" or "cuda").

    Returns:
        None
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            save_checkpoint(model, epoch, path="output/models/best_model.pth")
            print("Model improved, checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

