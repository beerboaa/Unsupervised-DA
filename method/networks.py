import torch
from torch import nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Function
from itertools import combinations
import numpy as np


def init_weights(net, gain=0.02):
    def init_func(layer):
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (classname.find('Conv') != -1):
            init.xavier_normal_(layer.weight.data,  gain)
            # init.normal_(layer.weight.data, 0.0, gain)
            if hasattr(layer, 'bias') and layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(layer.weight.data, 1.0, gain)
            init.constant_(layer.bias.data, 0.0)

    net.apply(init_func)
    print('Network initialized')

def set_parameter_requires_grad(model, unfreeze_layers):
    for param in model.parameters():
        param.requires_grad = False

    for layer in unfreeze_layers:
        for param in model[layer + 3].parameters(): # fine-tuning
            param.requires_grad = True


class ResNet50_encoder(nn.Module):
    """
    Resnet50 encoder
    """
    def __init__(self, unfreeze_layers=[4]):
        super(ResNet50_encoder, self).__init__()

        model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(model.children())[:-1])

        if unfreeze_layers is not None or len(unfreeze_layers) > 0:
            set_parameter_requires_grad(self.encoder, unfreeze_layers)

    def forward(self, input):
        bs = input.size(0)
        conv_out = self.encoder(input)

        return conv_out.view(bs, -1)

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
        self.fc1 = nn.Linear(50 * 5 * 5, 500)

    def forward(self, input):
        """Forward the LeNet."""

        bs = input.size(0)
        conv_out = self.encoder(input)
        return self.fc1(conv_out.view(bs, -1))

class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.fc1 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        # out = self.fc1(feat)
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc1(out)
        return out

class DigitDiscriminator(nn.Module):
    """
    Discriminator network for digits
    """
    def __init__(self):
        super(DigitDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(*[nn.Linear(500, 500),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(500, 500),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(500, 1)])

    def forward(self, input):
        return self.discriminator(input)

class Discriminator(nn.Module):
    """
    Domain discriminator
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(*[nn.Linear(2048, 512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(512, 512),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(512, 1)])

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


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if torch.cuda.is_available():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if torch.cuda.is_available(): classes = classes.cuda()
        labels = labels.long().unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def __call__(self, embeddings, labels):

        with torch.no_grad():
            triplets = self.get_triplets(embeddings, labels)
        if not isinstance(triplets, torch.Tensor):
            return None

        f_A = embeddings[triplets[:, 0]]
        f_P = embeddings[triplets[:, 1]]
        f_N = embeddings[triplets[:, 2]]

        ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
        an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_D - an_D + 1.)

        return losses.mean()

    def get_triplets(self, embeddings, y):

        margin = 1
        D = self.pdist(embeddings)
        D = D.cpu()

        y = y.cpu().data.numpy().ravel()
        trip = []

        for label in set(y):
            label_mask = (y == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            neg_ind = np.where(np.logical_not(label_mask))[0]

            ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
            ap = np.array(ap)

            ap_D = D[ap[:, 0], ap[:, 1]]

            # # GET HARD NEGATIVE
            # if np.random.rand() < 0.5:
            #   trip += get_neg_hard(neg_ind, hardest_negative,
            #                D, ap, ap_D, margin)
            # else:
            trip += self.get_neg_hard(neg_ind, self.random_neg, D, ap, ap_D, margin)

        if len(trip) == 0:
            return None
            # ap = ap[0]
            # trip.append([ap[0], ap[1], neg_ind[0]])

        trip = np.array(trip)

        return torch.LongTensor(trip)

    def pdist(self, vectors):
        D = -2 * vectors.mm(torch.t(vectors))
        D += vectors.pow(2).sum(dim=1).view(1, -1)
        D += vectors.pow(2).sum(dim=1).view(-1, 1)

        return D

    def get_neg_hard(self, neg_ind, select_func, D, ap, ap_D, margin):
        trip = []

        for ap_i, ap_di in zip(ap, ap_D):
            loss_values = (ap_di -
                           D[torch.LongTensor(np.array([ap_i[0]])),
                             torch.LongTensor(neg_ind)] + margin)

            loss_values = loss_values.data.cpu().numpy()
            neg_hard = select_func(loss_values)

            if neg_hard is not None:
                neg_hard = neg_ind[neg_hard]
                trip.append([ap_i[0], ap_i[1], neg_hard])

        return trip

    def random_neg(self, loss_values):
        neg_hards = np.where(loss_values > 0)[0]
        return np.random.choice(neg_hards) if len(neg_hards) > 0 else None






