# cnn_models.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import alexnet

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomResNet18, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        features = self.model.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.model.fc(features)
        return logits, features
    
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        features = self.model.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.model.fc(features)
        return logits, features

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomAlexNet, self).__init__()
        self.alexnet = alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        features = self.alexnet.features(x)
        features = self.alexnet.avgpool(features)
        features_flattened = torch.flatten(features, 1)
        penultimate_features = self.alexnet.classifier[:-1](features_flattened)
        logits = self.alexnet.classifier[-1](penultimate_features)
        return logits, penultimate_features

def load_cnn_model(model_type, model_weights=None,num_classes=2):
    """Loads the specified CNN model with optional pretrained weights."""
    if model_type == "resnet18":
        model = CustomResNet18(num_classes)
    elif model_type == "alexnet":
        model = CustomAlexNet(num_classes)
    elif model_type == "resnet50":
        model = CustomResNet50(num_classes)
    else:
        raise ValueError("Unsupported model type. Choose from 'resnet18', 'alexnet', 'resnet50'.")

    # Load custom weights if provided
    if model_weights:
        model.load_state_dict(torch.load(model_weights,weights_only=True))
    
    model.eval()  # Set to evaluation mode by default
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and test CNN model.")
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet18", "alexnet", "resnet50"], help="Type of CNN model to load.")
    parser.add_argument("--model_weights", type=str, help="Path to pretrained model weights.")
    args = parser.parse_args()

    # Load the model and print to verify
    model = load_cnn_model(args.model_type, args.model_weights)
    print(f"Loaded {args.model_type} model with weights from {args.model_weights if args.model_weights else 'default initialization'}")
