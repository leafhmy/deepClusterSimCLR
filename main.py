import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from get_args import parse_args
from load_data.dataloader import WBCDataset
from load_model.model_loader import load_deep_cluster, load_simclr
from cluster import clustering
from utils.update import *
from cluster.clustering import *


def train_dc(args, dc_model, sim_model,
             optimizer, criterion,
             deepcluster, dataloader, dataset, fd):
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
        dc_model.top_layer = None
        # model.classifier = nn.Sequential(*list(model.module.classifier.children())[:-1])
        dc_model.classifier = nn.Sequential(*list(dc_model.classifier.children())[:-1])

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
                                       dataset.imgs, from_simclr=True, confidence=args.confidence)

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

        # set last fully connected layer
        mlp = list(dc_model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        dc_model.classifier = nn.Sequential(*mlp)
        dc_model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        dc_model.top_layer.weight.data.normal_(0, 0.01)
        dc_model.top_layer.bias.data.zero_()
        dc_model.top_layer.cuda()

        loss = train_deepcluster(args, train_dataloader, dc_model, criterion, optimizer, epoch)
        print(f'epoch [{epoch}] loss: {loss} clustering_loss: {clustering_loss}')

        if epoch % 10 == 0:
            # 重新聚类再次训练SimCLR模型 设置间隔epoch
            print(f'epoch {epoch} clustering... updating SimCLR...')
            train_dataset = deep_cluster_high_confidence(args, dc_model, deepcluster, dataloader, dataset)
            # model.train()
            train_dataset = train_dataset.images
            sim_model = updata_simclr(args, sim_model, train_dataset, updata_epoch=20)
            print(f'updating SimCLR done')

    # save running checkpoint
    torch.save({'epoch': epoch + 1,
                'arch': args.arch_d,
                'state_dict': dc_model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(args.exp, 'checkpoint_dc_final_epoch{}.pth.tar').format(epoch + 1))


def main(args):
    dc_model, fd = load_deep_cluster(args)
    sim_model = load_simclr(args)

    WBC = WBCDataset(args)
    loader = WBC.get_data_loader()
    path2label = WBC.get_path2label()
    dataset = WBC.get_dataset()

    deepcluster = Kmeans(k=args.nmb_cluster)

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, dc_model.parameters()),
        lr=args.lr_d,
        momentum=args.momentum,
        weight_decay=10 ** args.wd_d,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    train_dc(args, dc_model, sim_model,
             optimizer, criterion, deepcluster, loader, dataset, fd)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '13, 14, 15'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # args.verbose = True
    # args.batch = 2
    # args.data = 'E:\dataset\segmentation_datasets\WBC_images/'
    # args.resume = 'D:\PythonProjects\deepcluster_checkpoint/checkpoint_0_alexnet.pth.tar'
    # args.resume_sim = 'D:\PythonProjects\deepcluster_checkpoints/checkpoint_0100.pth.tar'
    main(args)

"""
python main.py --verbose --batch 128 --faiss-gpu --confidence 0.3 --lr_s 0.05
"""


