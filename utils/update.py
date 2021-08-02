import os

import torch.nn as nn
from cluster.clustering import *
from load_data.dataloader import WBCDataset
from utils.simclr import SimCLR


def deep_cluster_high_confidence(args, model, deepcluster, dataloader, dataset):
    # 不注释会完蛋
    # # remove head
    # model.top_layer = None
    # # model.classifier = nn.Sequential(*list(model.module.classifier.children())[:-1])
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    # # get the features for the whole dataset
    features = compute_features(args, dataloader, model, len(dataset))

    # cluster the features
    if args.verbose:
        print('Cluster the features')
    clustering_loss = deepcluster.cluster(features, verbose=args.verbose, from_simclr=True)

    # assign pseudo-labels
    if args.verbose:
        print('Assign pseudo labels')
    train_dataset = cluster_assign(deepcluster.images_lists, dataset.imgs)

    return train_dataset


def updata_simclr(args, model, train_dataset, updata_epoch):
    """
    train_dataset: high confidence
    :param args:
    :param model:
    :param train_dataset:
    :param updata_epoch:
    :return:
    """
    loader = WBCDataset(args, train_dataset)
    train_dataset = loader.get_data_loader()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr_s, weight_decay=args.wd_s)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, updata_epoch=updata_epoch)

    return model


def train_deepcluster(args, loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr_d,
        weight_decay=10**args.wd_d,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    # save checkpoint
    if epoch > 0 and epoch % args.checkpoints == 0:
        path = os.path.join(
            args.exp,
            'checkpoints',
            'checkpoint_' + str(epoch) + '.pth_dc.tar',
        )
        if args.verbose:
            print('Save checkpoint at: {0}'.format(path))
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch_d,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict()
        }, path)

    return losses.avg