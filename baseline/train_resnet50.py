import argparse
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from resnet50 import ResNet50_encoder, ResNet50_classifier
from resnet50_data_loader import Dataset
from utils import save_model, init_model

def init_parser(parser):
    parser.add_argument('--data_dir', required=True, type=str, help='path to data directory containing train and test folders')
    parser.add_argument('--name', required=True, type=str, help='name of the task')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--restore_encoder', type=str, default=None, help='path to load model to continue training')
    parser.add_argument('--restore_classifier', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=1, help='which epoch to resume')
    parser.add_argument('--save_path', type=str, default=None, help='where to save the model')
    parser.add_argument('--num_classes', type=int, required=True, help='number of classes')
    parser.add_argument('--image_size', type=str, default=(224, 224), help='load image size')

    return parser

def train(encoder, classifier, train_loader, test_loader, opt):
    "Train ResNet50"

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # start training
    for epoch in range(opt.resume_epoch, opt.epoch + 1):
        for i, (images, labels) in enumerate(train_loader):

            # zero gradient for optimizer
            optimizer.zero_grad()

            # compute loss
            preds = classifier(encoder(images))
            loss = criterion(preds, labels.long())

            # optimize
            loss.backward()
            optimizer.step()

            # print step info
            if ((i + 1) % 100 == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              opt.epoch,
                              i + 1,
                              len(train_loader),
                              loss.data.item()))

        test_loss, test_acc, class_acc = test(encoder, classifier, test_loader, opt)
        print('End of epoch {}, loss={}, test_loss={}, test_acc={}'.format(epoch,
                                                                           loss.data.item(),
                                                                           test_loss,
                                                                           test_acc))
        print('Class accuracy:{}'.format(class_acc))

        with open(os.path.join(opt.save_path, 'loss_logs.txt'), 'a') as f:
            f.write('Epoch:{}, training loss:{}, test loss:{}, test acc:{}\n'.format(epoch,
                                                                                     loss.data.item(),
                                                                                     test_loss,
                                                                                     test_acc))
            f.write('Class accuracy:{}\n'.format(class_acc))

        if epoch % 5 == 0:
            save_model(encoder, os.path.join(opt.save_path, "ResNet50_encoder_{}.pt".format(epoch)))
            save_model(classifier, os.path.join(opt.save_path, "ResNet50_classifier_{}.pt".format(epoch)))


def test(encoder, classifier, test_loader, opt):
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    acc = 0
    loss = 0

    class_acc = np.zeros(opt.num_classes)
    class_total = np.zeros(opt.num_classes)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = classifier(encoder(images))
            loss += criterion(preds, labels.long()).data.item()

            pred_cls = torch.max(preds.item(), 1)[1]
            acc += (pred_cls == labels.long()).sum().item()
            c = (pred_cls == labels.long()).squeeze()

            for correct, label in zip(c.data.item(), labels.data.item()):
                if correct == 1:
                    class_acc[label] += 1
                class_total[label] += 1

    loss /= len(test_loader)
    acc /= len(test_loader.dataset)
    class_acc /= class_total

    return loss, acc, class_acc

if __name__=="__main__":
    parser =argparse.ArgumentParser()
    parser = init_parser(parser)
    opt = parser.parse_args()

    encoder = ResNet50_encoder()
    classifier = ResNet50_classifier(num_classes=opt.num_classes)
    if opt.restore_encoder != None and os.path.exists(opt.restore_encoder):
        encoder.load_state_dict(torch.load(opt.restore_encoder))
        encoder.restored = True
    if opt.restore_classifier != None and os.path.exists(opt.restore_classifier):
        classifier.load_state_dict(torch.load(opt.restore_classifier))
        classifier.restored = True

    if torch.cuda.is_available():
        cudnn.benchmark = True
        encoder.to('cuda')
        classifier.to('cuda')

    if opt.mode == 'train':
        dataset = Dataset(opt.data_dir, opt.mode)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True)
        dataset = Dataset(opt.data_dir, 'test')
        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        with open(os.path.join(opt.save_path, 'loss_logs.txt'), 'a') as f:
            f.write('Begin training\n')

        train(encoder, classifier, train_loader, test_loader, opt)

    else:
        dataset = Dataset(opt.data_dir, opt.mode)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True)
        test_loss, test_acc, class_acc = test(encoder, classifier, data_loader, opt)
        print('Test loss = {}, Test acc = {}'.format(test_loss, test_acc))
