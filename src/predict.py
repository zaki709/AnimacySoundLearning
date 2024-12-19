import torch
import torch.nn as nn
from torchvision.models import resnet34


def main():
    resnet_model = resnet34(pretrained=False)
    resnet_model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    resnet_model.fc = nn.Linear(512, 50)  # 50は分類クラス数

    # 保存済みパラメータをロード
    resnet_model.load_state_dict(torch.load("output/resnet_model.pth"))
    resnet_model.eval()  # 推論モードに設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet_model.to(device)


if __name__ == "__main__":
    main()
