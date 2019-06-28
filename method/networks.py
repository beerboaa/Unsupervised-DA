import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn.functional as F


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

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

    def forward(self, input):
        """Forward the LeNet."""

        bs = input.size(0)
        conv_out = self.encoder(input)
        return conv_out.view(bs, -1)

class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.fc1(feat)
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out

class DigitDiscriminator(nn.Module):
    """
    Discriminator network for digits
    """
    def __init__(self):
        super(DigitDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(*[nn.Linear(512, 512),
                                             nn.LeakyReLU(0.2),
                                             # nn.Linear(512, 512),
                                             # nn.LeakyReLU(0.2),
                                             nn.Linear(512, 1)])

    def forward(self, input):
        return self.discriminator(input)

class Discriminator(nn.Module):
    """
    Domain discriminator
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(*[nn.Linear(2048, 1024),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(1024, 1024),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(1024, 1)])

    def forward(self, input):
        return self.discriminator(input)

class GANLoss(nn.Module):
    """
    GANloss
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss





