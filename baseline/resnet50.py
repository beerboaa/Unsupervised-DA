from torchvision import models
import torch.nn as nn

class ResNet50_encoder(nn.Module):
    """
    ResNet50 only for features extractor
    """
    def __init__(self):
        super(ResNet50_encoder, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet50.children())[:-1])

    def forward(self, input):
        bs = input.size(0)
        features = self.encoder(input)
        features = features.view(bs, -1)

        return features

class ResNet50_classifier(nn.Module):
    """
    fully-connected layer for ResNet50
    """
    def __init__(self, num_classes):
        super(ResNet50_classifier, self).__init__()
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, input):
        output = self.fc(input)
        return output

