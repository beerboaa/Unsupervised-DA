import torch
import networks
from torch import nn


class UDAAN(nn.Module):
    def __init__(self, opt):
        super(UDAAN, self).__init__()
        self.share_encoder = opt.share_encoder
        self.use_center_loss = opt.use_center_loss
        self.threshold_T = opt.threshold_T

        # init encoder
        if opt.share_encoder:
            if opt.image_size == 32:
                self.encoder = networks.LeNetEncoder()
                networks.init_weights(self.encoder)
            else:
                self.encoder = networks.ResNet50_encoder(opt.unfreeze_layers)



        else:
            if opt.image_size == 32:
                self.encoder_s = networks.LeNetEncoder()
                self.encoder_t = networks.LeNetEncoder()
                networks.init_weights(self.encoder_s)
                networks.init_weights(self.encoder_t)
            else:
                self.encoder_s = networks.ResNet50_encoder(opt.unfreeze_layers)
                self.encoder_t = networks.ResNet50_encoder(opt.unfreeze_layers)



        # init discriminator and classifier
        if opt.image_size == 32:
            self.discriminator = networks.DigitDiscriminator()
            self.classifier = networks.LeNetClassifier()
        else:
            self.discriminator = networks.Discriminator()
            self.classifier = networks.Classifier(opt.num_classes)

        networks.init_weights(self.discriminator)
        networks.init_weights(self.classifier)

        if opt.use_center_loss:
            if opt.image_size == 32:
                self.center_loss = networks.CenterLoss(opt.num_classes, self.classifier.fc1.in_features)
            else:
                self.center_loss = networks.CenterLoss(opt.num_classes, self.classifier.fc.in_features)


    def forward(self, input, labels, domains):
        # consider 'source domain'
        center_loss = None
        if domains.sum().item() == 0:
            if self.share_encoder:
                features = self.encoder(input)
            else:
                features = self.encoder_s(input)

            label_prediction = self.classifier(features)
            domain_prediction = self.discriminator(features)

            if self.use_center_loss:
                center_loss = self.center_loss(features, labels)

            return label_prediction, domain_prediction, center_loss

        # consider 'target domain'
        else:
            if self.share_encoder:
                features = self.encoder(input)
            else:
                features = self.encoder_t(input)

            label_prediction = self.classifier(features)
            domain_prediction = self.discriminator(features)

            # pseudo labels
            if self.use_center_loss:
                prob, label_prediction = torch.max(nn.Softmax(dim=1)(label_prediction), 1)

                if len(prob[prob > self.threshold_T]) > 0:
                    center_loss = self.center_loss(features, label_prediction)


            return None, domain_prediction, center_loss









