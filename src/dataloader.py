import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split


class AnimacyDataset(Dataset):
    def __init__(self, csv_file, target_length=16000, sample_rate=16000, n_mels=128):
        self.data = pd.read_csv(csv_file, header=None)
        self.target_length = target_length  # 固定長に設定（例: 1秒の16kサンプル）
        self.n_mels = n_mels
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels
        )
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data.iloc[idx, 0], int(self.data.iloc[idx, 1])
        waveform, sr = torchaudio.load(file_path)

        # 複数チャンネルの場合、平均を計算してモノラル化
        if waveform.size(0) == 2:
            waveform = waveform.mean(keepdim=True, dim=0)

        # サンプリングレートの統一
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 音声を固定長に揃える(trim or 0-padding)
        if waveform.size(1) > self.target_length:
            center = waveform.size(1) // 2
            start = max(0, center - self.target_length // 2)
            end = start + self.target_length
            waveform = waveform[:, start:end]
        else:
            pad_length = self.target_length - waveform.size(1)
            left_pad = pad_length // 2
            right_pad = pad_length - left_pad
            waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))

        # メルスペクトログラム変換を適用
        melspectrogram = self.mel_transform(waveform)

        return melspectrogram, label


def get_dataloader(
    csv_file, batch_size=32, shuffle=True, target_length=32000, val_split=0.2
):
    dataset = AnimacyDataset(csv_file, target_length=target_length)
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
