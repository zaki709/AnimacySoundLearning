import torch
import torch.nn as nn
from torchvision import models


def initialize_model(model_name="resnet34",num_classes=2,pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"unsupported model name: {model_name}. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,num_classes)

    return model

# example usage
# model = initialize_model()
# model = initialize_model(model_name=*prefer model name*, num_classes=*number of classes*)
