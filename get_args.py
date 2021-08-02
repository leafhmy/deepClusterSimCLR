import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster+SimCLR')
    # DeepCluster
    parser.add_argument('--data', metavar='DIR', help='path to dataset',
                        # default='E:\dataset\segmentation_datasets\WBC_images/')
                        default='/home/st003/hmy/dataset/BloodCellSigned/segmentation_datasets/images/')
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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=8, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='/home/st003/hmy/DeepClusterSimCLR/checkpoints/checkpoint_0_alexnet.pth.tar',
                        type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=20,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=999, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--faiss-gpu', action='store_true', default=False, help='use faiss-gpu')
    parser.add_argument('--confidence', type=float, default=0.3)

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
    parser.add_argument('--resume_sim', type=str, default="/home/st003/hmy/SimCLR/runs/Apr16_19-08-13_gpu02/checkpoint_0100.pth.tar")

    return parser.parse_args()