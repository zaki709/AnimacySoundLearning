import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34


def train(train_loader, test_loader):
    resnet_model = resnet34(pretrained=True)
    resnet_model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # 最後の層の次元を今回のカテゴリ数に変更する
    resnet_model.fc = nn.Linear(512, 50)

    # GPU使います。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=2e-4)

    train_losses = []
    for epoch in range(50):
        train_losses = 0
        for data in train_loader:
            optimizer.zero_grad()
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            x = x.unsqueeze(1)  # チャネル数1を挿入
            out = resnet_model(x)
            loss = loss_function(out, y)
            loss.backward()
            optimizer.step()
            train_losses += loss.item()

        # 検証
        test_losses = 0
        actual_list, predict_list = [], []

        for data in test_loader:
            with torch.no_grad():
                x, y = data
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
                x = x.unsqueeze(1)
                out = resnet_model(x)
                loss = loss_function(out, y)
                _, y_pred = torch.max(out, 1)
                test_losses += loss.item()

                actual_list.append(y.cpu().numpy())
                predict_list.append(y_pred.cpu().numpy())

        actual_list = np.concatenate(actual_list)
        predict_list = np.concatenate(predict_list)
        accuracy = np.mean(actual_list == predict_list)

        # epoch毎の精度確認
        print(
            "epoch",
            epoch,
            "\t train_loss",
            train_losses,
            "\t test_loss",
            test_losses,
            "\t accuracy",
            accuracy,
        )
    return resnet_model
