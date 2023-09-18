import torch.utils.data as data
from PIL import Image
import numpy as np
from scipy.io import loadmat
from os import path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset


class DigitFiveDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(DigitFiveDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]

    def append(self, b ,c, d):
        self.data = np.concatenate((self.data, b.data, c.data, d.data), axis = 0)
        self.labels = np.concatenate((self.labels, b.labels, c.labels, d.labels), axis = 0)
        return self



def load_mnist(base_path):
    print("load mnist")
    mnist_data = loadmat(path.join(base_path, "dataset", "DigitFive", "mnist_data.mat"))
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    # turn to the 3 channel image with C*H*W
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train0 = mnist_train[inds]
    train_label0 = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train0[:20000]
    train_label = train_label0[:20000]
    mnist_val = mnist_train0[20000:25000]
    val_label = train_label0[20000:25000]
    mnist_test = mnist_test[:9000]
    test_label = test_label[:9000]
    return mnist_train, train_label, mnist_val, val_label, mnist_test, test_label


def load_mnist_m(base_path):
    print("load mnistm")
    mnistm_data = loadmat(path.join(base_path, "dataset", "DigitFive", "mnistm_with_label.mat"))
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train0 = mnistm_train[inds]
    train_label0 = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    mnistm_train = mnistm_train0[:20000]
    train_label = train_label0[:20000]
    mnistm_val = mnistm_train0[20000:25000]
    val_label = train_label0[20000:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    return mnistm_train, train_label, mnistm_val, val_label, mnistm_test, test_label


def load_svhn(base_path):
    print("load svhn")
    svhn_train_data = loadmat(path.join(base_path, "dataset", "DigitFive", "svhn_train_32x32.mat"))
    svhn_test_data = loadmat(path.join(base_path, "dataset", "DigitFive", "svhn_test_32x32.mat"))
    svhn_train = svhn_train_data['X']
    svhn_train = svhn_train.transpose(3, 2, 0, 1).astype(np.float32)

    svhn_test = svhn_test_data['X']
    svhn_test = svhn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = svhn_train_data["y"].reshape(-1)
    test_label = svhn_test_data["y"].reshape(-1)
    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train0 = svhn_train[inds]
    train_label0 = train_label[inds]
    svhn_train = svhn_train0[:20000]
    train_label = train_label0[:20000]
    svhn_val = svhn_train0[20000:25000]
    val_label = train_label0[20000:25000]
    svhn_test = svhn_test[:9000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return svhn_train, train_label, svhn_val, val_label, svhn_test, test_label


def load_syn(base_path):
    print("load syn train")
    syn_train_data = loadmat(path.join(base_path, "dataset", "DigitFive", "synth_train_32x32.mat"))
    print("load syn test")
    syn_test_data = loadmat(path.join(base_path, "dataset", "DigitFive", "synth_test_32x32.mat"))
    syn_train = syn_train_data["X"]
    syn_test = syn_test_data["X"]
    syn_train0 = syn_train.transpose(3, 2, 0, 1).astype(np.float32)
    syn_test = syn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label0 = syn_train_data["y"].reshape(-1)
    test_label = syn_test_data["y"].reshape(-1)
    syn_train = syn_train0[:20000]
    train_label = train_label0[:20000]
    syn_val = syn_train0[20000:25000]
    val_label = train_label0[20000:25000]
    syn_test = syn_test[:9000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return syn_train, train_label, syn_val, val_label, syn_test, test_label


def load_usps(base_path):
    print("load usps")
    usps_dataset = loadmat(path.join(base_path, "dataset", "DigitFive", "usps_28x28.mat"))
    usps_dataset = usps_dataset["dataset"]
    usps_train = usps_dataset[0][0]
    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0
    usps_test = usps_dataset[1][0]
    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0
    usps_train = usps_train * 255
    usps_test = usps_test * 255
    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    usps_train0 = np.tile(usps_train, (4, 1, 1, 1))
    train_label0 = np.tile(train_label,4)
    usps_train = usps_train0[:20000]
    train_label = train_label0[:20000]
    usps_val = usps_train0[20000:25000]
    val_label = train_label0[20000:25000]
    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    return usps_train, train_label, usps_val, val_label, usps_test, test_label

def get_dataset(data_func, base_path, transform):
    train_image, train_label, val_image, val_label, test_image, test_label = data_func(base_path)
    train_dataset = DigitFiveDataset(data=train_image, labels=train_label, transform=transform)
    val_dataset = DigitFiveDataset(data=val_image, labels=val_label, transform=transform)
    test_dataset = DigitFiveDataset(data=test_image, labels=test_label, transform=transform)
    return train_dataset, val_dataset, test_dataset

def merge_dataset(train_dataset1, val_dataset1, test_dataset1, train_dataset2, val_dataset2, test_dataset2, train_dataset3, val_dataset3, test_dataset3, train_dataset4, val_dataset4, test_dataset4, batch_size):
    all_train_datasets = [train_dataset1, train_dataset2, train_dataset3, train_dataset4]
    all_val_datasets = [val_dataset1, val_dataset2, val_dataset3, val_dataset4]
    all_test_datasets = [test_dataset1, test_dataset2, test_dataset3, test_dataset4]
    merged_train_dataset = ConcatDataset(all_train_datasets)
    merged_val_dataset = ConcatDataset(all_val_datasets)
    merged_test_dataset = ConcatDataset(all_test_datasets)
    train_loader = DataLoader(merged_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(merged_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(merged_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

def digit5_dataset_read(base_path, domain, batch_size, exp = False):
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if exp == False:
        if domain == "mnist":
            train_image, train_label, val_image, val_label, test_image, test_label = load_mnist(base_path)
        elif domain == "mnistm":
            train_image, train_label, val_image, val_label, test_image, test_label = load_mnist_m(base_path)
        elif domain == "svhn":
            train_image, train_label, val_image, val_label, test_image, test_label = load_svhn(base_path)
        elif domain == "syn":
            train_image, train_label, val_image, val_label, test_image, test_label = load_syn(base_path)
        elif domain == "usps":
            train_image, train_label, val_image, val_label, test_image, test_label = load_usps(base_path)
        else:
            raise NotImplementedError("Domain {} Not Implemented".format(domain))
        train_dataset = DigitFiveDataset(data=train_image, labels=train_label, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataset = DigitFiveDataset(data=val_image, labels=val_label, transform=transform)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = DigitFiveDataset(data=test_image, labels=test_label, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, val_loader, test_loader
    else:
        print("Source_Combine Model")
        train_dataset0, val_dataset0, test_dataset0 = get_dataset(load_mnist, base_path, transform)
        train_dataset1, val_dataset1, test_dataset1 = get_dataset(load_mnist_m, base_path, transform)
        train_dataset2, val_dataset2, test_dataset2 = get_dataset(load_svhn, base_path, transform)
        train_dataset3, val_dataset3, test_dataset3 = get_dataset(load_syn, base_path, transform)
        train_dataset4, val_dataset4, test_dataset4 = get_dataset(load_usps, base_path, transform)

        if domain == "mnist":
            train_loader, val_loader, test_loader = merge_dataset(train_dataset1, val_dataset1, test_dataset1, train_dataset2, val_dataset2, test_dataset2, train_dataset3, val_dataset3, test_dataset3, train_dataset4, val_dataset4, test_dataset4, batch_size)
            return train_loader, val_loader, test_loader
        elif domain == "mnistm":
            train_loader, val_loader, test_loader = merge_dataset(train_dataset0, val_dataset0, test_dataset0, train_dataset2, val_dataset2, test_dataset2, train_dataset3, val_dataset3, test_dataset3, train_dataset4, val_dataset4, test_dataset4, batch_size)
            return train_loader, val_loader, test_loader
        elif domain == "svhn":
            train_loader, val_loader, test_loader = merge_dataset(train_dataset1, val_dataset1, test_dataset1, train_dataset0, val_dataset0, test_dataset0, train_dataset3, val_dataset3, test_dataset3, train_dataset4, val_dataset4, test_dataset4, batch_size)
            return train_loader, val_loader, test_loader
        elif domain == "syn":
            train_loader, val_loader, test_loader = merge_dataset(train_dataset1, val_dataset1, test_dataset1, train_dataset2, val_dataset2, test_dataset2, train_dataset0, val_dataset0, test_dataset0, train_dataset4, val_dataset4, test_dataset4, batch_size)
            return train_loader, val_loader, test_loader
        elif domain == "usps":
            train_loader, val_loader, test_loader = merge_dataset(train_dataset1, val_dataset1, test_dataset1, train_dataset2, val_dataset2, test_dataset2, train_dataset3, val_dataset3, test_dataset3, train_dataset0, val_dataset0, test_dataset0, batch_size)
            return train_loader, val_loader, test_loader

