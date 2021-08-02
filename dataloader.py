"""
deep_cluster_high_confidence for SimCLR training
"""
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from SimCLR.data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from SimCLR.exceptions.exceptions import InvalidDatasetSelection


class WBCDataset:
    """
    from DeepCluster with high confidence
    """
    def __init__(self, images, n_views, regression=False):
        self.images = images
        self.n_views = n_views
        self.anno = './anno/'
        if not os.path.exists(self.anno):
            os.mkdir(self.anno)
        self._save_file_path()
        if not regression:
            self.trans = ContrastiveLearningViewGenerator(self._get_simclr_pipeline_transform(224), n_views)
        else:
            self.trans = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

    def _save_file_path(self):
        fpath = []
        labels = []

        for path, label in self.images:
            fpath.append(path)
            labels.append(label)

        with open(self.anno + 'train.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))

    def getDataLoader(self):
        train_loader = DatasetLoader(self.anno, 'train', self.trans)
        return train_loader


    def _get_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms


class DatasetLoader:
    def __init__(self, anno_dir, phase, data_transforms):
        self.anno = anno_dir
        self.data_transforms = data_transforms
        self.data = []
        with open(self.anno+phase+'.txt', 'r') as f:
            for item in f.readlines():
                img, label = item.split(' ')
                self.data.append((img, label.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = Image.open(img_path).convert('RGB')
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, int(label)
