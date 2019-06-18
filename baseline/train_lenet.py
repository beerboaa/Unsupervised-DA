import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from lenet import LeNetEncoder, LeNetClassifier
from data_loader import Dataset
from utils import save_model, init_model


def init_parser(parser):
    parser.add_argument('--data_dir', required=True, help='path to data directory containing train and test folders')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--model_path', type=str, default=None, help='path to load model')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='epoch')
    parser.add_argument('--restore', type=str, default=None, help='load model to continue training')
    parser.add_argument('--resume_epoch', type=int, default=1, 'which epoch to resume')
    parser.add_argument('--save_path', type=str, default=None, 'where to save the model')
    parser.add_argument('--image_size', type=str, default=(32, 32), help='load image size')

    return parser

def train(encoder, classifier, train_loader, test_loader, opt):
    "Train LeNet"

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=0.001, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()

    # start training
    for epoch in range(opt.resume_epoch, opt.epoch + 1):
        for i, (images, labels) in enumerate(train_loader):

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize
            loss.backward()
            optimizer.step()

            # print step info
            if ((i + 1) % 100 == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              opt.epoch,
                              step + 1,
                              len(train_loader),
                              loss.data[0]))

        test_loss, test_acc = test(encoder, classifier, test_loader, opt)
        print('End of epoch {}, loss={}, test_loss={}, test_acc={}'.format(epoch,
                                                                          loss.data[0],
                                                                          test_loss,
                                                                          test_acc))
        with open(os.path.join(opt.save_path, 'loss_logs.txt'), 'a') as f:
            f.write('Epoch:{}, training loss:{}, test loss:{}, test acc:{}\n'.format(epoch,
                                                                                   loss.data[0],
                                                                                   test_loss,
                                                                                   test_acc))
        if epoch % 5 == 0:
            save_model(encoder, "LeNet_encoder_{}.pt".format(epoch))
            save_model(classifier, "LeNet_classifier_{}.pt".format(epoch))



def test(encoder, classifier, test_loader, opt):
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    acc = 0
    loss =0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = classifier(encoder(images))
            loss += criterion(preds, labels).data[0]

            pred_cls = torch.max(preds.data, 1)
            acc += (pred_cls == labels).sum().item()

    loss /= len(test_loader)
    acc /= len(test_loader.dataset)

    return loss, acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = init_parser(parser)
    opt = parser.parse_args()

    encoder = init_model(net=LeNetEncoder(), restore=opt.restore)
    classifier = init_model(net=LeNetClassifier(), restore=opt.restore)

    if opt.mode == 'train':
        dataset = Dataset(opt.data_dir, opt.mode, opt.image_size)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True)
        dataset = Dataset(opt.data_dir, 'test', opt.image_size)
        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True)

        with open(os.path.join(opt.save_path, 'loss_logs.txt'), 'a') as f:
            f.write('Begin training\n')

        train(encoder, classifier, train_loader, test_loader, opt)

    else:
        dataset = Dataset(opt.data_dir, opt.mode)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True)

        test_loss, test_acc = test(encoder, classifier, data_loader, opt)
        print('Test loss = {}, Test acc = {}'.format(test_loss, test_acc))

