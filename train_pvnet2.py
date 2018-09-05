# -*- coding: utf-8 -*-
import config
from utils import meter
from torch import nn
from torch import optim
from models import PVNet2, PVNet2_v4
from torch.utils.data import DataLoader
from datasets import *


def train(train_loader, net, criterion, optimizer, epoch):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    net.train()

    for i, (views, pcs, labels) in enumerate(train_loader):
        batch_time.reset()
        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(pcs, views)  # bz x C x H x W
        loss = criterion(preds, labels)

        prec.add(preds.detach(), labels.detach())
        losses.add(loss.item())  # batchsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Loss {losses.value()[0]:.4f} \t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'prec at epoch {epoch}: {prec.value(1)} ')


def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)

    # testing mode
    net.eval()

    for i, (views, pcs, labels) in enumerate(val_loader):
        batch_time.reset()

        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(pcs, views)  # bz x C x H x W

        prec.add(preds.data, labels.data)

        prec.add(preds.data, labels.data)

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'mean class accuracy at epoch {epoch}: {prec.value(1)} ')
    return prec.value(1)


def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict, osp.join(config.pv_net.ckpt_record_folder, f'epoch{epoch}_{prec1:.2f}.pth'))


def save_ckpt(epoch, epoch_pc, epoch_all, best_prec1, net, optimizer_pc, optimizer_all, training_conf=config.pv_net):
    ckpt = dict(
        epoch=epoch,
        epoch_pc=epoch_pc,
        epoch_all=epoch_all,
        best_prec1=best_prec1,
        model=net.state_dict(),
        optimizer_pc=optimizer_pc.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
        training_conf=training_conf
    )
    torch.save(ckpt, config.pv_net.ckpt_file)


def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    train_dataset = pc_view_data(config.pv_net.pc_root,
                                 config.pv_net.view_root,
                                 status=STATUS_TRAIN,
                                 base_model_name=config.base_model_name)
    val_dataset = pc_view_data(config.pv_net.pc_root,
                               config.pv_net.view_root,
                               status=STATUS_TEST,
                               base_model_name=config.base_model_name)

    train_loader = DataLoader(train_dataset, batch_size=config.pv_net.train.batch_sz,
                              num_workers=config.num_workers,shuffle = True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.pv_net.train.batch_sz,
                            num_workers=config.num_workers,shuffle=True, drop_last=True)

    best_prec1 = 0
    resume_epoch = 0

    epoch_pc_view = 0
    epoch_pc = 0

    # create model
    net = PVNet2()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)

    # optimizer
    fc_param = [{'params': v} for k, v in net.named_parameters() if 'fusion' in k]
    if config.pv_net.train.optim == 'Adam':
        optimizer_fc = optim.Adam(fc_param, config.pv_net.train.fc_lr,
                                  weight_decay=config.pv_net.train.weight_decay)

        optimizer_all = optim.Adam(net.parameters(), config.pv_net.train.all_lr,
                                   weight_decay=config.pv_net.train.weight_decay)
    elif config.pv_net.train.optim == 'SGD':
        optimizer_fc = optim.SGD(fc_param, config.pv_net.train.fc_lr,
                                 momentum=config.pv_net.train.momentum,
                                 weight_decay=config.pv_net.train.weight_decay)

        optimizer_all = optim.SGD(net.parameters(), config.pv_net.train.all_lr,
                                  momentum=config.pv_net.train.momentum,
                                  weight_decay=config.pv_net.train.weight_decay)
    else:
        raise NotImplementedError
    print(f'use {config.pv_net.train.optim} optimizer')

    #pc 0.001       1 epoch        e  down 0.6
    #all 0.0001     1.5 epoch      e  down 0.1

    if config.pv_net.train.resume:
        print(f'loading pretrained model from {config.pv_net.ckpt_file}')
        checkpoint = torch.load(config.pv_net.ckpt_file)
        net.module.load_state_dict(checkpoint['model'])
        optimizer_fc.load_state_dict(checkpoint['optimizer_pc'])
        optimizer_all.load_state_dict(checkpoint['optimizer_all'])
        best_prec1 = checkpoint['best_prec1']
        epoch_pc_view = checkpoint['epoch_pc_view']
        epoch_pc = checkpoint['epoch_pc']
        if config.pv_net.train.resume_epoch is not None:
            resume_epoch = config.pv_net.train.resume_epoch
        else:
            resume_epoch = checkpoint['epoch'] + 1

    lr_scheduler_fc = torch.optim.lr_scheduler.StepLR(optimizer_fc, 5, 0.3)
    lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, 5, 0.3)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    for epoch in range(resume_epoch, config.pv_net.train.max_epoch):

        # for p in net.module.parameters():
        #     p.requires_grad = True

        #
        if epoch < 11:
            epoch_pc += 1
            lr_scheduler_fc.step(epoch=epoch_pc)
            train(train_loader, net, criterion, optimizer_fc, epoch)
        else:
            epoch_pc_view += 1
            lr_scheduler_all.step(epoch=epoch_pc_view)
            train(train_loader, net, criterion, optimizer_all, epoch)


        with torch.no_grad():
            prec1 = validate(val_loader, net, epoch)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            save_ckpt(epoch, epoch_pc, epoch_pc_view, best_prec1, net, optimizer_fc, optimizer_all)

        # save_record(epoch, prec1, net.module)
        print('curr accuracy: ', prec1)
        print('best accuracy: ', best_prec1)

    print('Train Finished!')


if __name__ == '__main__':
    main()

