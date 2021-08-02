import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils.gaussian_blur import GaussianBlur
from utils.view_generator import ContrastiveLearningViewGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class WBCDataset:
    def __init__(self, args, train_dataset=None):
        self.updata_simclr = True if train_dataset else False
        self.args = args
        self.dataset = train_dataset if train_dataset is not None else args.data
        self.anno = './anno/'
        if not os.path.exists(self.anno):
            os.mkdir(self.anno)
        if self.updata_simclr:
            self._create_anno_v2()
            self.data_transforms = self.get_trans(trans_simclr=True)
        else:
            self._create_anno_v1()
            self.data_transforms = self.get_trans(trans_simclr=False)
        self._create_loader()
        self.path2label = self._Loader.get_path2label()

    def get_dataset(self):
        self.trans = self.get_trans(trans_simclr=False)
        return datasets.ImageFolder(self.dataset, transform=transforms.Compose(self.trans))


    def get_trans(self, trans_simclr):
        if trans_simclr:
            trans = ContrastiveLearningViewGenerator(self._get_simclr_pipeline_transform(224), self.args.n_views)

        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        return trans

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

    def _create_anno_v1(self):
        fpath = []
        labels = []

        for index, d in enumerate(os.listdir(self.args.data)):
            fd = os.path.join(self.args.data, d)
            label = index
            for i in os.listdir(fd):
                fp = os.path.join(fd, i)
                fpath.append(fp)
                labels.append(label)

        with open(self.anno + 'train.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))

    def _create_anno_v2(self):
        """
        for high confidence
        :return:
        """
        fpath = []
        labels = []

        for path, label in self.dataset:
            fpath.append(path)
            labels.append(label)

        with open(self.anno + 'train_pesudo.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))

    def _create_loader(self):
        if self.updata_simclr:
            self._Loader = DatasetLoader(self.anno, 'train_pesudo', self.data_transforms)
        else:
            self._Loader = DatasetLoader(self.anno, 'train', self.data_transforms)

        self._loader = DataLoaderX(self._Loader,
                                  batch_size=self.args.batch,
                                  num_workers=self.args.workers,
                                  shuffle=False)

    def get_data_loader(self):
        return self._loader

    def get_path2label(self):
        return self.path2label


class DatasetLoader:
    def __init__(self, anno_dir, phase, data_transforms):
        self.anno = anno_dir
        self.data_transforms = data_transforms
        self.data = []
        with open(self.anno+phase+'.txt', 'r') as f:
            for item in f.readlines():
                img, label = item.split(' ')
                self.data.append((img, int(label.strip())))

    def __len__(self):
        return len(self.data)

    def get_path2label(self):
        return self.data

    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = Image.open(img_path).convert('RGB')
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label
