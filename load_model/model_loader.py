import models
from models.resnet_simclr import ResNetSimCLR
import torch
import torch.backends.cudnn as cudnn
import os


def load_deep_cluster(args):
    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch_d))
    model = models.__dict__[args.arch_d]()
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    # model = models.resnet18()
    model.features = torch.nn.DataParallel(model.features)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # remove top_layer parameters from checkpoint
        for key in list(checkpoint['state_dict'].keys()):
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        model.load_state_dict(checkpoint['state_dict'])

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    return model, fd


def load_simclr(args):
    model = ResNetSimCLR(base_model=args.arch_s, out_dim=args.out_dim)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    return model
