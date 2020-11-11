from torchvision.datasets import ImageFolder
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import torchvision.transforms.functional as F
from autoaugment import ImageNetPolicy


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    ImageNetPolicy(),
    transforms.ToTensor(),
    normalize,
])

val_transformer_ImageNet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

def test_transform(args):
    if args.test_mode == 'single':
        test_transformer_ImageNet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        test_transformer_ImageNet = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224, vertical_flip=False),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])

    return test_transformer_ImageNet



class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.num_classes = len(set(self.labels))

    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloaders(data_dir, ratio, batchsize=256, args=None):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)

    train_inputs, val_inputs = [], []
    train_labels, val_labels = [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio)
        num_sample_val = int(len(data) * (1-ratio))
        num_val_index = num_sample_train + num_sample_val

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)

    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet), 
                                  batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet), 
                                batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True)
    loader = {}
    loader['train'] = train_dataloader
    loader['val'] = val_dataloader

    return loader


def get_dataloaders(data_dir, args):
    loader = fetch_dataloaders(os.path.join(data_dir, 'train'), args.ratio, args.batch_size, args)
    test_loader = torch.utils.data.DataLoader(
        ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform(args)), 
        batch_size=100, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    loader['test'] = test_loader
    return loader


if __name__ == '__main__':
    data_dir = '/home/user/hhd/finegrained/train/'
    loader = fetch_dataloaders(data_dir, 0.9, 64)
    print(loader['train'].dataset.num_classes)
    print(len(loader['train'].dataset))
    print(len(loader['val'].dataset))
