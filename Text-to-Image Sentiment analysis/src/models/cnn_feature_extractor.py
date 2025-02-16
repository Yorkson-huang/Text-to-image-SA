import torch
import torchvision.models as models
from torch import nn

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(resnet.fc.in_features, feature_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)