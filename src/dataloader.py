import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class AnimalSoundDataset(Dataset):
    def __init__(self, csv_file, transform=None, fixed_length=16000):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.fixed_length = fixed_length  # 固定長に設定（例: 1秒の16kサンプル）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data.iloc[idx, 0], int(self.data.iloc[idx, 1])
        waveform, sample_rate = torchaudio.load(file_path)

        # 複数チャンネルの場合、平均を計算してモノラル化
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0)

        # 音声を固定長に揃える
        waveform = self._fix_length(waveform)

        # メルスペクトログラム変換を適用
        if self.transform:
            waveform = self.transform(waveform)  # [n_mels, time_steps]

        # ResNetに対応する4次元形式に変換: [1, n_mels, time_steps]
        waveform = waveform.unsqueeze(
            0
        )  # チャンネル次元を追加 -> [1, n_mels, time_steps]
        return waveform, label

    def _fix_length(self, waveform):
        """ゼロパディングまたは切り取りで固定長に揃える"""
        num_samples = self.fixed_length
        if waveform.shape[0] > num_samples:
            waveform = waveform[:num_samples]
        else:
            padding = num_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform


def collate_fn(batch):
    """カスタムコラート関数でバッチ内のテンソルサイズを揃える"""
    waveforms = torch.stack(
        [item[0] for item in batch]
    )  # [batch_size, 1, n_mels, time_steps]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels


def get_dataloader(
    csv_file, batch_size=32, shuffle=True, transform=None, fixed_length=16000
):
    dataset = AnimalSoundDataset(
        csv_file, transform=transform, fixed_length=fixed_length
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader

