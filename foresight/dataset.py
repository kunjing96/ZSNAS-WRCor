import torch, numpy as np, math, PIL
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

from .imagenet16 import *
from .autoaugment import ImageNetPolicy


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):

    def __init__(self, alphastd, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_dataloaders(train_batch_size, test_batch_size, dataset, num_workers, cutout=-1, datadir='_dataset'):

    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
        size, pad = 32, 4
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
        size, pad = 32, 4
    elif dataset == 'ImageNet16-120':
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
        size, pad = 16, 2
    elif dataset == 'ImageNet1k':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        crop_image_size = 320
        input_image_crop = 0.875
        resize_image_size = int(math.ceil(crop_image_size / input_image_crop))
    else:
        raise TypeError("Unknow dataset : {:}".format(dataset))

    if dataset != 'ImageNet1k':
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=pad),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
		]
        if cutout > 0: lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])
    else:
        lists = [
            transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomCrop(crop_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            Lighting(0.1),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
            transforms.RandomErasing(),
        ]
        train_transform = transforms.Compose(lists)

        test_transform  = transforms.Compose([
            transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(crop_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])

    if dataset == 'cifar10':
        train_dataset = CIFAR10(datadir, True, train_transform, download=True)
        test_dataset = CIFAR10(datadir, False, test_transform, download=True)
        assert len(train_dataset) == 50000 and len(test_dataset) == 10000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(datadir, True, train_transform, download=True)
        test_dataset = CIFAR100(datadir, False, test_transform, download=True)
        assert len(train_dataset) == 50000 and len(test_dataset) == 10000
    elif dataset == 'ImageNet16-120':
        train_dataset = ImageNet16(os.path.join(datadir, 'ImageNet16'), True , train_transform, 120)
        test_dataset  = ImageNet16(os.path.join(datadir, 'ImageNet16'), False, test_transform , 120)
        assert len(train_dataset) == 151700 and len(test_dataset) == 6000
    elif dataset == 'ImageNet1k':
        train_dataset = ImageFolder(os.path.join(datadir, 'ImageNet1k', 'train'), train_transform)
        test_dataset  = ImageFolder(os.path.join(datadir, 'ImageNet1k', 'val'),   test_transform)
        assert len(train_dataset) == 1281167 and len(test_dataset) == 50000
    else:
        raise ValueError('There are no more cifars or imagenets.')

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader

