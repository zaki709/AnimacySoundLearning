import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataloader import ESC50Data
from train import train


def main():
    meta_df = pd.read_csv("data/meta/esc50.csv")
    train_df, test_df = train_test_split(meta_df, train_size=0.8)

    train_data = ESC50Data("data/audio/", train_df, "filename", "category")
    test_data = ESC50Data("data/audio/", test_df, "filename", "category")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    model = train(train_loader, test_loader)
    torch.save(model.state_dict(), "output/resnet_model.pth")


if __name__ == "__main__":
    main()
