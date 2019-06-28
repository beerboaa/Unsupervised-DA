import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models

def init_weights(net, gain=0.02):
    def init_func(layer):
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.xavier_normal_(layer.weight.data, gain=gain)

            if hasattr(layer, 'bias') and layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(layer.weight.data, 1.0, gain)
            init.constant_(layer.bias.data, 0.0)

    net.apply(init_func)
    print('Network initialized')

def set_parameter_requires_grad(model, freeze_layers):
    for param in model.parameters():
        param.requires_grad = False

    for layer in freeze_layers:
        for param in model[layer].parameters(): # fine-tuning
            param.requires_grad = True


class ResNet50_encoder(nn.Module):
    """
    Resnet50 encoder
    """
    def __init__(self, freeze_layers=[4]):
        super(ResNet50_encoder, self).__init__()

        model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(model.children())[:-1])

        if freeze_layers is not None or len(freeze_layers) > 0:
            set_parameter_requires_grad(self.encoder, freeze_layers)

    def forward(self, input):
        return self.encoder(input)

class Classifier(nn.Module):
    """
    fully-connected layers for image classification
    """
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, input):
        return self.fc(input)

class LeNetEncoder




