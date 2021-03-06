import argparse
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from models import UDAAN
from networks import GANLoss, CenterLoss
from data_loader import TrainDataset, TestDataset

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True, type=str, help='path to data directory containing train and test folders')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_center', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_discriminator', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=200, help='epoch')
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
    parser.add_argument('--decay_step', type=int, default=40)
    parser.add_argument('--unfreeze_layers', type=list, default=[4], help='which layer to fine tune')
    parser.add_argument('--copy_epoch', type=int, default=30, help='which epoch to start training target networks')
    parser.add_argument('--use_triplet_loss', action='store_true', help='use triplet loss')
    parser.add_argument('--gen_path', type=str, default=None, help='use translated images for training')


    opt = parser.parse_args()

    return opt

def init_optimizer(model, opt):

    optimizers = {}

    if opt.share_encoder:
        if opt.image_size == 32:
            optimizers['encoder'] = optim.Adam(model.encoder.parameters(), lr=opt.lr_encoder, betas=(0.5, 0.999))
        else:
            optimizers['encoder'] = optim.SGD(model.encoder.parameters(), lr=opt.lr_encoder, weight_decay=5e-04, momentum=0.9)
    else:
        if opt.image_size == 32:
            optimizers['encoder_s'] = optim.Adam(model.encoder_s.parameters(), lr=opt.lr_encoder, betas=(0.5, 0.999))
            optimizers['encoder_t'] = optim.Adam(model.encoder_t.parameters(), lr=opt.lr_encoder * 0.2, betas=(0.5, 0.999))
        else:
            optimizers['encoder_s'] = optim.SGD(model.encoder_s.parameters(), lr=opt.lr_encoder, weight_decay=5e-04, momentum=0.9)
            optimizers['encoder_t'] = optim.SGD(model.encoder_t.parameters(), lr=opt.lr_encoder * 0.2, weight_decay=5e-04, momentum=0.9)
    if opt.use_center_loss:
        optimizers['center_loss'] = optim.SGD(model.center_loss.parameters(), lr=opt.lr_center, momentum=0.9)

    if opt.image_size == 32:
        optimizers['classifier'] = optim.Adam(model.classifier.parameters(), lr=opt.lr_classifier, betas=(0.5, 0.999))
        optimizers['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=opt.lr_discriminator, betas=(0.5, 0.999))
    else:
        optimizers['classifier'] = optim.SGD(model.classifier.parameters(), lr=opt.lr_classifier, weight_decay=5e-04, momentum=0.9)
        optimizers['discriminator'] = optim.SGD(model.discriminator.parameters(), lr=opt.lr_discriminator, weight_decay=5e-04, momentum=0.9)
    return optimizers

def init_scheduler(optimizers, opt):
    schedulers = {}

    if opt.share_encoder:
        schedulers['encoder'] = optim.lr_scheduler.StepLR(optimizers['encoder'], step_size=opt.decay_step, gamma=0.5)
    else:
        schedulers['encoder_s'] = optim.lr_scheduler.StepLR(optimizers['encoder_s'], step_size=opt.decay_step, gamma=0.5)
        schedulers['encoder_t'] = optim.lr_scheduler.StepLR(optimizers['encoder_t'], step_size=opt.decay_step, gamma=0.5)
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
    criterions['CELoss'] = nn.CrossEntropyLoss().to('cuda') if torch.cuda.is_available() else nn.CrossEntropyLoss()

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
    schedulers = init_scheduler(optimizers, opt)

    test_acc_max = 0.0
    # start training
    for epoch in range(1, opt.epoch + 1):
        # step the scheduler
        step_scheduler(schedulers)

        if epoch < 30:
            beta1 = 0.001
            beta2 = 0

        elif epoch < 60:
            beta1 = 0.002
            beta2 = 0.002

        else:
            beta1 = 0.01
            beta2 = 0.01


        for i, data in enumerate(train_loader):
            # p = float(i + epoch * len(train_loader) / opt.epoch / len(train_loader))
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels, target_images, target_labels, gen_images, gen_labels = data
            if torch.cuda.is_available():
                source_images = source_images.to('cuda')
                source_labels = source_labels.to('cuda')
                source_domains = torch.zeros_like(source_labels).to('cuda')
                target_images = target_images.to('cuda')
                target_labels = target_labels.to('cuda')
                target_domains = torch.ones_like(source_labels).to('cuda')
                if opt.gen_path is not None:
                    gen_images = gen_images.to('cuda')
                    gen_labels = gen_labels.to('cuda')
                    gen_domains = (torch.ones_like(source_labels) * (-1)).to('cuda')

            # Source
            source_label_preds, source_domain_preds, source_disc_loss = \
                                model(source_images, source_labels, source_domains)

            criterions = init_criterion(opt)

            loss_D_source = criterions['GANLoss'](source_domain_preds, True) * opt.alpha_d
            loss_E_source = criterions['GANLoss'](source_domain_preds, False) * opt.alpha_d
            loss_C_source = criterions['CELoss'](source_label_preds, source_labels.long()) * opt.alpha_c

            set_zero_grad(optimizers, optimizers.keys())
            if opt.use_center_loss:
                source_disc_loss *= beta1

                source_disc_loss.backward(retain_graph=True)
                for param in model.center_loss.parameters():
                    param.grad.data *= (1. / (beta1))
                nn.utils.clip_grad_value_(model.center_loss.parameters(), 5.0)
                optimizers['center_loss'].step()

            elif opt.use_triplet_loss:
                if isinstance(source_disc_loss, torch.Tensor):
                    source_disc_loss *= beta1
                    source_disc_loss.backward(retain_graph=True)

            if opt.share_encoder or epoch >= opt.copy_epoch:
                (loss_C_source + loss_E_source).backward(retain_graph=True)
            else:
                loss_C_source.backward()

            nn.utils.clip_grad_value_(model.classifier.parameters(), 5.0)
            optimizers['classifier'].step()
            if opt.share_encoder:
                nn.utils.clip_grad_value_(model.encoder.parameters(), 5.0)
                optimizers['encoder'].step()
            else:
                nn.utils.clip_grad_value_(model.encoder_s.parameters(), 5.0)
                optimizers['encoder_s'].step()

            if opt.share_encoder or epoch >= opt.copy_epoch:
                set_zero_grad(optimizers, optimizers.keys())
                loss_D_source.backward()
                nn.utils.clip_grad_value_(model.discriminator.parameters(), 5.0)
                optimizers['discriminator'].step()

            # Target
            target_label_preds, target_domain_preds, target_disc_loss = \
                                model(target_images, target_labels, target_domains)

            loss_D_target = criterions['GANLoss'](target_domain_preds, False) * opt.alpha_d
            loss_E_target = criterions['GANLoss'](target_domain_preds, True) * opt.alpha_d

            set_zero_grad(optimizers, optimizers.keys())
            if opt.use_center_loss and beta2 != 0 and target_disc_loss is not None:
                target_disc_loss *= beta2

                target_disc_loss.backward(retain_graph=True)
                for param in model.center_loss.parameters():
                    param.grad.data *= (1. / (beta2))
                nn.utils.clip_grad_value_(model.center_loss.parameters(), 5.0)
                optimizers['center_loss'].step()

            elif opt.use_triplet_loss and beta2 != 0 and target_disc_loss is not None:

                if isinstance(target_disc_loss, torch.Tensor):
                    target_disc_loss *= beta1
                    target_disc_loss.backward(retain_graph=True)

            if opt.share_encoder or epoch >= opt.copy_epoch:
                if epoch==opt.copy_epoch and not opt.share_encoder:
                    model.encoder_t.load_state_dict(model.encoder_s.state_dict())
                # set_zero_grad(optimizers, optimizers.keys())
                loss_E_target.backward(retain_graph=True)
                if opt.share_encoder:
                    nn.utils.clip_grad_value_(model.encoder.parameters(), 5.0)
                    optimizers['encoder'].step()
                else:
                    nn.utils.clip_grad_value_(model.encoder_t.parameters(), 5.0)
                    optimizers['encoder_t'].step()

                set_zero_grad(optimizers, optimizers.keys())
                loss_D_target.backward()
                nn.utils.clip_grad_value_(model.discriminator.parameters(), 5.0)
                optimizers['discriminator'].step()

            # Gen path
            if opt.gen_path is not None:
                gen_label_preds, gen_domain_preds, gen_disc_loss = \
                    model(gen_images, gen_labels, gen_domains)

                loss_D_gen = criterions['GANLoss'](gen_domain_preds, False) * opt.alpha_d
                loss_E_gen = criterions['GANLoss'](gen_domain_preds, True) * opt.alpha_d
                loss_C_gen = criterions['CELoss'](gen_label_preds, gen_labels.long()) * opt.alpha_c

                set_zero_grad(optimizers, optimizers.keys())
                if opt.use_center_loss and beta2 != 0 and gen_disc_loss is not None:
                    gen_disc_loss *= beta2

                    gen_disc_loss.backward(retain_graph=True)
                    for param in model.center_loss.parameters():
                        param.grad.data *= (1. / (beta2))
                    nn.utils.clip_grad_value_(model.center_loss.parameters(), 5.0)
                    optimizers['center_loss'].step()

                elif opt.use_triplet_loss and beta2 != 0 and gen_disc_loss is not None:

                    if isinstance(gen_disc_loss, torch.Tensor):
                        gen_disc_loss *= beta1
                        gen_disc_loss.backward(retain_graph=True)

                if opt.share_encoder or epoch >= opt.copy_epoch:
                    (loss_C_gen + loss_E_gen).backward(retain_graph=True)
                else:
                    loss_C_gen.backward()

                nn.utils.clip_grad_value_(model.classifier.parameters(), 5.0)
                optimizers['classifier'].step()

                if opt.share_encoder or epoch >= opt.copy_epoch:

                    # set_zero_grad(optimizers, optimizers.keys())

                    if opt.share_encoder:
                        nn.utils.clip_grad_value_(model.encoder.parameters(), 5.0)
                        optimizers['encoder'].step()
                    else:
                        nn.utils.clip_grad_value_(model.encoder_t.parameters(), 5.0)
                        optimizers['encoder_t'].step()

                    set_zero_grad(optimizers, optimizers.keys())
                    loss_D_gen.backward()
                    nn.utils.clip_grad_value_(model.discriminator.parameters(), 5.0)
                    optimizers['discriminator'].step()
            #
            # #==========================================================================================
            #
            if opt.use_center_loss or opt.use_triplet_loss:
                # print step info
                if ((i + 1) % 200 == 0):
                    print("Epoch [{}/{}] Step [{}/{}]:".format(epoch, opt.epoch, i + 1, len(train_loader)))
                    if source_disc_loss is not None:
                        print('loss_D_source={}, loss_C_source={}, loss_disc_source={}'.
                           format(loss_D_source.data.item(), loss_C_source.data.item(), source_disc_loss.data.item()))
                    if target_disc_loss is not None:
                        print('loss_D_target={}, loss_disc_target={}'.format(loss_D_target.data.item(), target_disc_loss.data.item()))
            else:
                if ((i + 1) % 200 == 0):
                    print("Epoch [{}/{}] Step [{}/{}]:".format(epoch, opt.epoch, i + 1, len(train_loader)))
                    print('loss_D_source={}, loss_C_source={}'.
                          format(loss_D_source.data.item(), loss_C_source.data.item() ))
                    print('loss_D_target={}'.format(loss_D_target.data.item()))

        test_loss, test_acc, class_acc = test(model, test_loader, criterions, opt)
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

        if test_acc > test_acc_max:
            test_acc_max = test_acc
            if opt.share_encoder:
                torch.save(model.encoder.state_dict(), os.path.join(opt.save_path, 'Encoder_{}.pt'.format(epoch)))
            else:
                torch.save(model.encoder_t.state_dict(), os.path.join(opt.save_path, 'Encoder_t_{}.pt'.format(epoch)))
            print("save best model to : {}".format(opt.save_path))


def test(model, test_loader, criterions, opt):
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


    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')
            if opt.share_encoder:
                preds = model.classifier(model.encoder(images))
            else:
                preds = model.classifier(model.encoder_t(images))
            loss += criterions['CELoss'](preds, labels.long()).data.item()

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
        cudnn.benchmark = True
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


