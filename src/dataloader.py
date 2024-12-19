import os

import librosa
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


# メルスペクトログラムを画像に変換する
class ESC50Data(Dataset):
    def __init__(self, base_path, df, in_col, out_col):
        self.df = df
        self.data = []  # 音源データをメルスペクトログラム（画像）に変換して格納する用
        self.labels = []  # 各データのカテゴリー情報を格納する
        self.category2id = {}
        self.id2category = {}
        self.categories = list(
            sorted(df[out_col].unique())
        )  # 正解ラベル格納用（５０ラベル）
        # ラベルをIDに変換する辞書を作成
        for i, category in enumerate(self.categories):
            self.category2id[category] = i
            self.id2category[i] = category

        # メタ情報ファイルからファイル名を取得し、wavデータを1件ずつ画像に変換していく
        for row in tqdm(range(len(df))):
            row = df.iloc[row]
            file_path = os.path.join(base_path, row[in_col])
            waveform, sr = librosa.load(file_path)

            # メルスペクトログラムを取得してデシベルスケールに変換
            feature_melspec = librosa.feature.melspectrogram(y=waveform, sr=sr)
            feature_melspec_db = librosa.power_to_db(feature_melspec, ref=np.max)

            self.data.append(feature_melspec_db)
            self.labels.append(self.category2id[row["category"]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
