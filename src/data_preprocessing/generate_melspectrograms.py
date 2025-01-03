import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# from utils import create_directory


def audio_to_melspectrogram(filepath, output_dir, sr=22050, n_mels=128, fmax=8000):
    y, sr = librosa.load(filepath, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)

    output_file = os.path.join(
        output_dir, os.path.basename(filepath).replace(".wav", ".png")
    )
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=fmax)
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()
    return output_file


if __name__ == "__main__":
    ret = []
    for root, dirs, files in os.walk("data/raw"):
        for file in files:
            ret.append(os.path.join(root, file))
            print(root)
    print(ret[:10])
