import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast
import faiss
import numpy as np
from PIL import Image
from DeepCluster.run_deep_cluster import run_deep_cluster, compute_features
import DeepCluster.models as models
from dataloader import WBCDataset
from SimCLR.simclr import SimCLR
from SimCLR.models.resnet_simclr import ResNetSimCLR
from DeepCluster.run_deep_cluster import train
from DeepCluster.util import AverageMeter, Logger, UnifLabelSampler
os.environ['CUDA_VISIBLE_DEVICES'] = '10'


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model

class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        self.images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            self.images.append((path, pseudolabel))
        return self.images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

def cluster_assign(images_lists, dataset, confidence=0.4, from_simclr=False):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    if not from_simclr:
        len_min_cluster = int(min([len(l) for l in images_lists]) * confidence)
        for i in range(len(images_lists)):
            images_lists[i] = images_lists[i][:len_min_cluster]
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

def preprocess_features(npdata, pca=256, from_simclr=False):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    if not from_simclr:
        mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)

        # L2 normalization
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False, use_gpu=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    if use_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
    else:
        index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    distance, I = index.search(x, 1)
    clus_index2dis = [(clus_index, dis) for clus_index, dis in zip(I, distance)]

    # losses = faiss.vector_to_array(clus.obj)  # this option was replaced. The fix is:
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return clus_index2dis, losses[-1]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False, use_gpu=False, from_simclr=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, from_simclr=from_simclr)

        # cluster the data
        clus_index2dis, loss = run_kmeans(xb, self.k, verbose, use_gpu=use_gpu)
        self.images_lists = [[] for i in range(self.k)]

        for i in range(len(data)):
            self.images_lists[clus_index2dis[i][0][0]].append((clus_index2dis[i][1][0], i))

        if not from_simclr:
            for clus in self.images_lists:
                clus.sort(key=lambda t: t[0])

        for i in range(self.k):
            self.images_lists[i] = [index for _, index in self.images_lists[i]]

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster+SimCLR')
    # DeepCluster
    parser.add_argument('--data', metavar='DIR', help='path to dataset',
                        default='E:\dataset\segmentation_datasets\WBC_images/')
                        # default='/home/st003/hmy/dataset/BloodCellSigned/segmentation_datasets/images/')
    parser.add_argument('--arch_d', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'resnet'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--arch_s', metavar='ARCH', default='resnet18')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=5,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr_d', default=0.05, type=float,
                        help='learning rate for DeepCluster(default: 0.05)')
    parser.add_argument('--lr_s', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate for SimCLR')
    parser.add_argument('--wd_d', default=-5, type=float,
                        help='weight decay pow for DeepCluster(default: -5)')
    parser.add_argument('--wd_s', default=1e-4, type=float,
                        help='weight decay (default: 1e-4) for SimCLR')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=8, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='D:\PythonProjects\deepcluster_checkpoint/checkpoint_0_alexnet.pth.tar',
                        type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=20,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=999, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='./checkpoints/', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--faiss-gpu', action='store_true', default=False, help='use faiss-gpu')

    # SimCLR
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--out_dim', default=256, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=10, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--sim_model', type=str, default=' ')

    return parser.parse_args()


def deep_cluster_with_simclr(deepcluster, dataloader, dataset, simclr_model):
    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch_d))
    model = models.__dict__[args.arch_d](sobel=args.sobel)
    # model = models.resnet18()
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    # train DeepCluster
    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr_d,
        momentum=args.momentum,
        weight_decay=10 ** args.wd_d,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in list(checkpoint['state_dict']):
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    sim_model = ResNetSimCLR(base_model=args.arch_s, out_dim=args.out_dim).to(device)
    checkpoint = torch.load(simclr_model, map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    sim_model.load_state_dict(state_dict, strict=False)

    for epoch in range(args.start_epoch, args.epochs):
        feature_matrix = np.empty((1, 256))
        for images, _ in dataloader:
            images = images.to(args.device)

            with autocast(enabled=args.fp16_precision):
                features = sim_model(images)
                features = features.detach().cpu().numpy()
                feature_matrix = np.vstack((feature_matrix, features))

        feature_matrix = feature_matrix[1:]

        # remove head
        model.top_layer = None
        # model.classifier = nn.Sequential(*list(model.module.classifier.children())[:-1])
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # from simclr features
        features = feature_matrix

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose, from_simclr=True)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        train_dataset = cluster_assign(deepcluster.images_lists,
                                       dataset.imgs, from_simclr=True)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # define loss function
        criterion = nn.CrossEntropyLoss().cuda()

        # for epoch in range(args.start_epoch, args.epochs):
        loss = train(args, train_dataloader, model, criterion, optimizer, epoch)
        print(f'epoch [{epoch}] loss: {loss} clustering_loss: {clustering_loss}')

        # 重新聚类再次训练SimCLR模型 设置间隔epoch
        print(f'epoch {epoch} clustering... updating SimCLR...')


        if epoch % args.checkpoints == 0:
            # save running checkpoint
            torch.save({'epoch': epoch + 1,
                        'arch': args.arch_d,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(args.exp, 'checkpoint_dc_epoch{}.pth.tar'.format(epoch)))



def deep_cluster_high_confidence(deepcluster, dataloader, dataset):
    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch_d))
    # model = models.__dict__[args.arch_d](sobel=args.sobel)
    model = models.resnet18()
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr_d,
        momentum=args.momentum,
        weight_decay=10 ** args.wd_d,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in list(checkpoint['state_dict']):
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # remove head
    model.top_layer = None
    # model.classifier = nn.Sequential(*list(model.module.classifier.children())[:-1])
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features = compute_features(args, dataloader, model, len(dataset))

    # cluster the features
    if args.verbose:
        print('Cluster the features')
    clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

    # assign pseudo-labels
    if args.verbose:
        print('Assign pseudo labels')
    train_dataset = cluster_assign(deepcluster.images_lists,
                                              dataset.imgs)

    return train_dataset

def train_simclr(train_dataset):
    loader = WBCDataset(train_dataset, args.n_views)
    train_dataset = loader.getDataLoader()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch_s, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr_s, weight_decay=args.wd_s)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


def main(args):
    print(f"train DeepCluster model: {args.arch_d}")
    run_deep_cluster(args)
    print(f"train DeepCluster done")
    a = input()

    # clustering algorithm to use
    deepcluster = Kmeans(k=5)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    train_dataset = deep_cluster_high_confidence(deepcluster, dataloader, dataset)
    train_dataset = train_dataset.images

    # train_simclr(train_dataset)

    loader = WBCDataset(train_dataset, args.n_views)
    train_dataset = loader.getDataLoader()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # simclr_model = 'D:\PythonProjects\simclr_model/checkpoint_0200.pth.tar'
    simclr_model = args.sim_model
    deep_cluster_with_simclr(deepcluster, dataloader, dataset, simclr_model)



if __name__ == '__main__':
    # deep_cluster_checkpoint = 'DeepCluster/checkpoints/checkpoint.pth.tar'
    deep_cluster_checkpoint = 'D:\PythonProjects\deepcluster_checkpoint/checkpoint_0_alexnet.pth.tar'
    args = parse_args()
    main(args)

"""
python main.py --epoch 100 --batch 64 --resume '
/home/st003/hmy/DeepClusterSimCLR/checkpoints/checkpoint_0_alexnet.pth.tar' --faiss-gpu --sim_m
odel '/home/st003/hmy/DeepClusterSimCLR/runs/Apr08_21-36-06_gpu01/checkpoint_0200_simclr.pth.ta
r' --data '/home/st003/hmy/dataset/BloodCellSigned/segmentation_datasets/images/'
"""


