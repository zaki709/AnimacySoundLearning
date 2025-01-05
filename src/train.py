import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as transforms
from sklearn.metrics import accuracy_score

from dataloader import get_dataloader
from resnet_model import AudioClassifier

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
CSV_FILE = "data/metadata/dataset_metadata.csv"

# Transform: Convert waveform to Mel Spectrogram
transform = transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
)

# Prepare DataLoader
train_loader = get_dataloader(CSV_FILE, batch_size=BATCH_SIZE, transform=transform)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
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
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}"
    )

torch.save(model.state_dict(), "output/models/animal_sound_classifier.pth")

