import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from dataloader import get_dataloader
from logger import Logger
from notification.notify import send_email
from resnet_model import AudioClassifier

# Configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
CSV_FILE = "data/metadata/dataset_metadata.csv"
PATIENCE = 5
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
TRAIN_ID = (datetime.datetime.now(JST)).strftime("%Y%m%d%H%M%S")

# Prepare DataLoader
train_loader, val_loader = get_dataloader(CSV_FILE, batch_size=BATCH_SIZE)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early Stopping variables
best_val_loss = np.inf
best_model_path = "output/models"

# Logger
logger = Logger(TRAIN_ID)
params = {
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LEARNING_RATE": LEARNING_RATE,
    "PATIENCE": PATIENCE,
    "MODEL": model.uname(),
}
logger.log_params(prefix="Training Parameters", params=params)


# train loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    train_log = (
        f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
    print(train_log)
    logger.log(train_log)

    # Validation
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    val_accuracy = accuracy_score(y_true, y_pred)
    val_log = f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    print(val_log)
    logger.log(val_log)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = f"{best_model_path}/best_model_{TRAIN_ID}.pth"
        torch.save(model.state_dict(), save_path)
        val_loss_log = f"Validation loss improved. Model saved to {save_path}."
        print(val_loss_log)
        logger.log(val_loss_log)
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        no_improvement_log = f"No improvement for {no_improvement_epochs} epochs."
        print(no_improvement_log)
        logger.log(no_improvement_log)

    if no_improvement_epochs >= PATIENCE:
        early_stop_log = "Early stopping triggered."
        print(early_stop_log)
        logger.log(early_stop_log)
        break

print("Training complete.")
logger.log("Training complete.")
send_email("train is done.")
