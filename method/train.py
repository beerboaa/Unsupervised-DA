import argparse
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from models import UDAAN
from networks import GANLoss
from data_loader import TrainDataset, TestDataset

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True, type=str, help='path to data directory containing train and test folders')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_center', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_discriminator', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=300, help='epoch')
    parser.add_argument('--num_classes', type=int, required=True, help='number of classes')
    parser.add_argument('--image_size', type=int, required=True, help='input size')
    parser.add_argument('--share_encoder', action='store_true', help='share or unshare encoder')
    parser.add_argument('--use_center_loss', action='store_true', help='use center loss')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='lsgan | vanilla')
    parser.add_argument('--direction', type=str, required=True, help='e.g. AtoB, MNISTtoUSPS')
    parser.add_argument('--save_path', type=str, required=True, help='path to save log')
    parser.add_argument('--alpha_c', type=float, default=10)
    parser.add_argument('--alpha_d', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.002)
    parser.add_argument('--threshold_T', type=float, default=0.9)
    parser.add_argument('--decay_step', type=int, default=50)

    opt = parser.parse_args()

    return opt

def init_optimizer(model, opt):
    enc_parameters = []
    optimizers = {}

    # if opt.share_encoder:
    #     enc_parameters.append({'params': model.encoder.parameters(), 'lr': opt.lr_encoder})
    # else:
    #     enc_parameters.append({'params': model.encoder_s.parameters(), 'lr': opt.lr_encoder})
    #     enc_parameters.append({'params': model.encoder_t.parameters(), 'lr': opt.lr_encoder})
    #
    # if opt.use_center_loss:
    #     cen_parameters = {'params': model.center_loss.parameters(), 'lr': opt.lr_center}
    #
    # cls_parameters = {'params': model.classifier.parameters(), 'lr': opt.lr_classifier}
    # dis_parameters = {'params': model.discriminator.parameters(), 'lr': opt.lr_discriminator}

    if opt.share_encoder:
        optimizers['encoder'] = optim.Adam(model.encoder.parameters(), lr=opt.lr_encoder, betas=(0.5, 0.999))
    else:
        optimizers['encoder_s'] = optim.Adam(model.encoder_s.parameters(), lr=opt.lr_encoder, betas=(0.5, 0.999))
        optimizers['encoder_t'] = optim.Adam(model.encoder_t.parameters(), lr=opt.lr_encoder, betas=(0.5, 0.999))
    if opt.use_center_loss:
        optimizers['center_loss'] = optim.Adam(model.center_loss.parameters(), lr=opt.lr_center, betas=(0.5, 0.999))
    optimizers['classifier'] = optim.Adam(model.classifier.parameters(), lr=opt.lr_classifier, betas=(0.5, 0.999))
    optimizers['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=opt.lr_discriminator, betas=(0.5, 0.999))
    # optimizer = optim.SGD(parameters, lr=opt.lr_encoder, momentum=0.9)

    return optimizers

def init_scheduler(optimizers, opt):
    schedulers = {}

    if opt.share_encoder:
        schedulers['encoder'] = optim.lr_scheduler.StepLR(optimizers['encoder'], step_size=opt.decay_step, gamma=0.5)
    else:
        schedulers['encoder_s'] = optim.lr_scheduler.StepLR(optimizers['encoder_s'], step_size=opt.decay_step, gamma=0.1)
        schedulers['encoder_t'] = optim.lr_scheduler.StepLR(optimizers['encoder_t'], step_size=opt.decay_step, gamma=0.1)
    # if opt.use_center_loss:
    #     # schedulers['center_loss'] = optim.lr_scheduler.StepLR(optimizers['center_loss'], step_size=opt.decay_step, gamma=0.1)
    schedulers['classifier'] = optim.lr_scheduler.StepLR(optimizers['classifier'], step_size=opt.decay_step, gamma=0.5)
    schedulers['discriminator'] = optim.lr_scheduler.StepLR(optimizers['discriminator'], step_size=opt.decay_step, gamma=0.5)

    return schedulers

def step_scheduler(schedulers):
    for k in schedulers.keys():
        schedulers[k].step()

def set_zero_grad(optimizers, optimizer_keys):
    for key in optimizer_keys:
        optimizers[key].zero_grad()

def init_criterion(opt):
    criterions = {}

    criterions['GANLoss'] = GANLoss(opt.gan_mode).to('cuda') if torch.cuda.is_available() else GANLoss(opt.gan_mode)
    criterions['CELoss'] = nn.CrossEntropyLoss()

    return criterions

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train(model, train_loader, test_loader, opt):
    # set train state
    if opt.share_encoder:
        model.encoder.train()
    else:
        model.encoder_s.train()
        model.encoder_t.train()

    model.classifier.train()
    model.discriminator.train()

    if opt.use_center_loss:
        model.center_loss.train()

    # setup criterion
    optimizers = init_optimizer(model, opt)
    criterions = init_criterion(opt)
    schedulers = init_scheduler(optimizers, opt)

    # start training
    for epoch in range(1, opt.epoch + 1):
        # step the scheduler
        step_scheduler(schedulers)

        if epoch < 5:
            beta1 = 0.001
            beta2 = 0

        elif epoch < 10:
            beta1 = 0.002
            beta2 = 0.002

        else:
            beta1 = 0.02
            beta2 = 0.02


        for i, data in enumerate(train_loader):
            # p = float(i + epoch * len(train_loader) / opt.epoch / len(train_loader))
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels, target_images, target_labels = data
            if torch.cuda.is_available():
                source_images = source_images.to('cuda')
                source_labels = source_labels.to('cuda')
                source_domains = torch.zeros_like(source_labels).to('cuda')
                target_images = target_images.to('cuda')
                target_labels = target_labels.to('cuda')
                target_domains = torch.ones_like(source_labels).to('cuda')

            # Source
            source_label_preds, source_domain_preds, source_center_loss = \
                                model(source_images, source_labels, source_domains)

            loss_D_source = criterions['GANLoss'](source_domain_preds, True) * opt.alpha_d
            loss_E_source = criterions['GANLoss'](source_domain_preds, False) * opt.alpha_d
            loss_C_source = criterions['CELoss'](source_label_preds, source_labels.long()) * opt.alpha_c

            loss_source = loss_D_source + loss_C_source

            if source_center_loss is not None:
                source_center_loss *= beta1
                loss_source += source_center_loss

                set_zero_grad(optimizers, optimizers.keys())
                source_center_loss.backward(retain_graph=True)
                for param in model.center_loss.parameters():
                    param.grad.data *= (1. / beta1)
                optimizers['center_loss'].step()

            # set_zero_grad(optimizers, optimizers.keys())
            (loss_C_source + loss_E_source).backward(retain_graph=True)
            optimizers['classifier'].step()
            if opt.share_encoder:
                optimizers['encoder'].step()
            else:
                optimizers['encoder_s'].step()
                optimizers['encoder_t'].step()

            # set_zero_grad(optimizers, optimizers.keys())
            # loss_E_source.backward(retain_graph=True)
            # if opt.share_encoder:
            #     optimizers['encoder'].step()
            # else:
            #     optimizers['encoder_s'].step()
            #     optimizers['encoder_t'].step()

            set_zero_grad(optimizers, optimizers.keys())
            loss_D_source.backward()
            optimizers['discriminator'].step()

            # # Target
            target_label_preds, target_domain_preds, target_center_loss = \
                                model(target_images, target_labels, target_domains)

            loss_D_target = criterions['GANLoss'](target_domain_preds, False) * opt.alpha_d
            loss_E_target = criterions['GANLoss'](target_domain_preds, True) * opt.alpha_d
            loss_target = loss_D_target
            if target_center_loss is not None and beta2 != 0 :
                target_center_loss *= beta2
                loss_target += target_center_loss

                set_zero_grad(optimizers, optimizers.keys())
                target_center_loss.backward(retain_graph=True)
                for param in model.center_loss.parameters():
                    param.grad.data *= (1. / (beta2))
                optimizers['center_loss'].step()

            set_zero_grad(optimizers, optimizers.keys())
            loss_E_target.backward(retain_graph=True)
            if opt.share_encoder:
                optimizers['encoder'].step()
            else:
                optimizers['encoder_s'].step()
                optimizers['encoder_t'].step()

            set_zero_grad(optimizers, optimizers.keys())
            loss_D_target.backward()
            optimizers['discriminator'].step()

            # optimizer.zero_grad()
            # loss = loss_source + loss_target
            # loss.backward()
            # for param in model.center_loss.parameters():
            #     param.grad.data *= (1. / beta1)
            # # loss_target.backward()
            # optimizer.step()
            # #
            if opt.use_center_loss:
                # print step info
                if ((i + 1) % 100 == 0):
                    print("Epoch [{}/{}] Step [{}/{}]:".format(epoch, opt.epoch, i + 1, len(train_loader)))
                    print('loss_D_source={}, loss_C_source={}, loss_center_source={}'.
                           format(loss_D_source.data.item(), loss_C_source.data.item(), source_center_loss.data.item()))
                    if target_center_loss is not None:
                        print('loss_D_target={}, loss_center_targe={}'.format(loss_D_target.data.item(), target_center_loss.data.item()))
            else:
                if ((i + 1) % 100 == 0):
                    print("Epoch [{}/{}] Step [{}/{}]:".format(epoch, opt.epoch, i + 1, len(train_loader)))
                    print('loss_D_source={}, loss_C_source={}'.
                          format(loss_D_source.data.item(), loss_C_source.data.item() ))
                    print('loss_D_target={}'.format(loss_D_target.data.item()))

        test_loss, test_acc, class_acc = test(model, test_loader, opt)
        print('End of epoch {}, loss={}, test_loss={}, test_acc={}'.format(epoch,
                                                                           loss_C_source.data.item(),
                                                                           test_loss,
                                                                           test_acc))
        print('Class accuracy:{}'.format(class_acc))

        with open(os.path.join(opt.save_path, 'loss_logs.txt'), 'a') as f:
            f.write('Epoch:{}, training loss:{}, test loss:{}, test acc:{}\n'.format(epoch,
                                                                                     loss_C_source.data.item(),
                                                                                     test_loss,
                                                                                     test_acc))
            f.write('Class accuracy:{}\n'.format(class_acc))


def test(model, test_loader, opt):
    # set eval state
    if opt.share_encoder:
        model.encoder.eval()
    else:
        model.encoder_s.eval()
        model.encoder_t.eval()

    model.classifier.eval()
    model.discriminator.eval()

    if opt.use_center_loss:
        model.center_loss.eval()

    acc = 0
    loss = 0

    class_acc = np.zeros(opt.num_classes)
    class_total = np.zeros(opt.num_classes)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')
            if opt.share_encoder:
                preds = model.classifier(model.encoder(images))
            else:
                preds = model.classifier(model.encoder_t(images))
            loss += criterion(preds, labels.long()).data.item()

            pred_cls = torch.max(preds, 1)[1]
            acc += (pred_cls == labels.long()).sum().item()
            c = (pred_cls == labels.long())

            for correct, label in zip(c, labels):
                if correct == 1:
                    class_acc[label] += 1
                class_total[label] += 1
        loss *= opt.alpha_c
        loss /= len(test_loader)
        acc /= len(test_loader.dataset)
        class_acc /= class_total

        return loss, acc, class_acc


if __name__=="__main__":
    opt = init_parser()
    model = UDAAN(opt)

    if torch.cuda.is_available():
        model.to('cuda')

    source, target = opt.direction.split('to')
    dataset = TrainDataset(source, opt)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True)
    dataset = TestDataset(target, opt)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    with open(os.path.join(opt.save_path, 'loss_logs.txt'), 'a') as f:
        f.write('Begin training\n')

    train(model, train_loader, test_loader, opt)


