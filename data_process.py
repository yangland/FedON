
from torch import nn
from torchvision import datasets
import torchvision.transforms as T
import os
import sys
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from torch.utils.data import TensorDataset
import scipy.io
import random
import numpy as np
import torch
import logging
from datasets import load_dataset as huggingface_load_dataset
import math
from collections import Counter
from more_itertools import chunked
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from torchvision.datasets import ImageFolder
import os

last_percent_reported = None

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def data_process(dataset, image_size, grayscale=False, percentage=1.0):
    resize = image_size[1]
    
    vanilla_list = [T.ToTensor()]  # Converts to [C,H,W] and scales to [0,1]
    
    # Standard normalization values for common datasets
    if dataset in ['mnist', 'fmnist', 'omniglot', 'semeion', 'emnist', 'emnist-letters', 'kmnist']:
        # Grayscale datasets: mean=0.1307, std=0.3081 (MNIST/FMNIST standard)
        transform_list = vanilla_list + [
            T.Resize([resize, resize]),
            T.Normalize(mean=[0.1307], std=[0.3081])  # MNIST/FMNIST stats
        ]
    else:
        # RGB datasets (e.g., CIFAR10: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        transform_list = vanilla_list + [
            T.Resize([resize, resize]),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR10 stats
        ]
    
    # Optional: Convert RGB to grayscale (only for RGB datasets)
    if grayscale and dataset not in ['mnist', 'fmnist', 'omniglot', 'semeion', 'emnist', 'emnist-letters', 'kmnist']:
        transform_list = [T.Grayscale(num_output_channels=1)] + transform_list
        # Update normalization for grayscale-converted images (use CIFAR10's global mean/std)
        transform_list[-1] = T.Normalize(mean=[0.5], std=[0.5])  # Approximate global stats
    
    transform = T.Compose(transform_list)
    
    # Load datasets
    if dataset == 'mnist':
        ds_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        ds_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif dataset == 'fmnist':
        ds_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        ds_test = datasets.FashionMNIST('./data', train= False, download=True, transform=transform)    
    elif dataset == 'cifar10':
        ds_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        ds_test = datasets.CIFAR10('./data', train= False, download=True, transform=transform)
    elif dataset == 'cifar100':
        ds_train = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        ds_test = datasets.CIFAR100('./data', train= False, download=True, transform=transform)
    elif dataset == 'cifar10-64':
        ds_train = datasets.ImageFolder(os.path.join("data/cifar10-64", "train"), transform=transform)
        ds_test = datasets.ImageFolder(os.path.join("data/cifar10-64", "test"), transform=transform)
    elif dataset == 'svhn':
        ds_train = datasets.SVHN('./data/SVHN', split = 'train', download=True, transform=transform)
        ds_test = datasets.SVHN('./data/SVHN', split = 'test', download=True, transform=transform)
    elif dataset == 'imagenet10':
        imagenette_dir = './data/imagenette2-160'  # Change to -320 or -224 if you prefer

        if not os.path.exists(imagenette_dir):
            raise RuntimeError(f"Imagenette directory not found at {imagenette_dir}. "
                            f"Please download it using the provided method below.")

        ds_train = ImageFolder(os.path.join(imagenette_dir, 'train'), transform=transform)
        ds_test = ImageFolder(os.path.join(imagenette_dir, 'val'), transform=transform)
    elif dataset == 'omniglot':
        orig_set = datasets.Omniglot('./data',  download=True, transform=transform)   
        
        n_total = len(orig_set)  # total number of examples
        n_test = int(0.15 * n_total)  # take ~10% for test
        ds_test = torch.utils.data.Subset(orig_set, range(n_test))  # take first 10%
        ds_train = torch.utils.data.Subset(orig_set, range(n_test, n_total))  # take the rest  
    elif dataset == 'semeion':
        # resize to 28 for FL
        orig_set = datasets.SEMEION('./data',  download=True, transform=transform)   
        
        n_total = len(orig_set)  # total number of examples
        n_test = int(0.15 * n_total)  # take ~10% for test
        ds_test = torch.utils.data.Subset(orig_set, range(n_test))  # take first 10%
        ds_train = torch.utils.data.Subset(orig_set, range(n_test, n_total))  # take the rest  
    elif dataset == 'emnist-letters':
        ds_train = datasets.EMNIST('./data', split = 'letters', train=True, download=True, transform=transform)
        ds_test = datasets.EMNIST('./data', split = 'letters', train=False, download=True, transform=transform)
    elif dataset == "imagenette":
        # ds_train = datasets.Imagenette(root='./data/imagenette2-320/train', split='train', size="320px", download=False, transform=transform)
        # ds_test = datasets.Imagenette(root='./data/imagenette2-320/test', split='val', size="320px", download=False, transform=transform)
        # ds_train = CachedDataset(ds_train)
        # ds_test = CachedDataset(ds_test)
        ds_train = load_or_cache_imagenette(split='train', transform=transform)
        ds_test = load_or_cache_imagenette(split='test', transform=transform)
    elif dataset == 'emnist-digits':
        ds_train = datasets.EMNIST('./data', split = 'digits', train=True, download=True, transform=transform)
        ds_test = datasets.EMNIST('./data', split = 'digits', train=False, download=True, transform=transform)
    elif dataset == 'emnist':
        ds_train = datasets.EMNIST('./data', split = 'balanced', train=True, download=True, transform=transform)
        ds_test = datasets.EMNIST('./data', split = 'balanced', train=False, download=True, transform=transform)
    elif dataset == 'kmnist':
        ds_train = datasets.KMNIST('./data', train=True, download=True, transform=transform)
        ds_test = datasets.KMNIST('./data', train=False, download=True, transform=transform) 
    elif dataset == 'oxford-pet':
        ds_train = datasets.OxfordIIITPet('./data', split = 'trainval', target_types="category", download=True, transform = transform)      
        ds_test = datasets.OxfordIIITPet('./data', split = 'test', target_types="category", download=True, transform = transform)
    elif dataset == 'dtd':
        # print("dtd transform", transform)
        ds_train = datasets.DTD('./data', split='train', download=True, transform=transform)
        ds_test = datasets.DTD('./data', split='test', download=True, transform=transform)
    elif dataset == 'places365':
        # ds_train = datasets.Places365('./data', split="train-standard", small=True, download=False, transform=transform)
        # ds_train = create_fl_trust_root_dataset(ds_train, fltrust_root_iid=False, p_degree=0.002304, fl_class_num=434, sample_size=60000)
        ds_test = datasets.Places365('./data', split="val", small=True, download=True, transform=transform)
        ds_train = ds_test
    elif dataset == 'celeba':
        ds_train = datasets.CelebA('./data', split="train", target_type="attr", download=True, transform=transform)
        ds_test = datasets.CelebA('./data', split="test", target_type="attr", download=True, transform=transform)
    elif dataset == 'GTSRB':
        ds_train = datasets.GTSRB('./data', split="train", download=True, transform=transform)
        ds_test = datasets.GTSRB('./data', split="test", download=True, transform=transform)
    elif dataset == 'eurosat':
        ds_train = datasets.EuroSAT('./data',download=True, transform=transform)
        ds_test = ds_train
    elif dataset == 'stanford-dogs':
        ds_train = huggingface_load_dataset("amaye15/stanford-dogs", split='train')
        ds_test = huggingface_load_dataset("amaye15/stanford-dogs", split='test')
        
        # this step is very slow
        ds_train = to_torch_ds(ds_train, transform)
        ds_test = to_torch_ds(ds_train, transform)
    elif dataset == 'imagenette':
        ds = huggingface_load_dataset("johnowhitaker/imagenette2-320")['train']
        ds_torch = to_torch_ds(ds, transform)
        random.shuffle(ds_torch)
        n_total = len(ds_torch)
        n_test = int(0.10* n_total)
        ds_test = torch.utils.data.Subset(ds_torch, range(n_test))
        ds_train = torch.utils.data.Subset(ds_torch, range(n_test, n_total)) 
        # print(ds_train[0][0])
        # print(ds_train[0][1])
    elif dataset == 'ch':
        #  biologists by focusing on histology tiles from patients with colorectal cancer
        ds = huggingface_load_dataset("dpdl-benchmark/colorectal_histology")['train'] 
        ds_torch = to_torch_ds(ds, transform)
        random.shuffle(ds_torch)
        n_total = len(ds_torch)
        n_test = int(0.10* n_total)
        ds_test = torch.utils.data.Subset(ds_torch, range(n_test))
        ds_train = torch.utils.data.Subset(ds_torch, range(n_test, n_total)) 
    elif dataset == 'spirit':
        file_dir = "data/sprite"
        transform = T.Compose([
            T.ToTensor(),  # from [0,255] to range [0.0,1.0]
            lambda x: 2*x - 1       # range [-1,1]
        ])
        target_transform = lambda x: torch.from_numpy(x).to(torch.float32)
        orig_set = SpriteDataset(os.path.join(file_dir), transform, target_transform)

        n_total = len(orig_set)
        n_test = int(0.10 * n_total)
        ds_test = torch.utils.data.Subset(orig_set, range(n_test))
        ds_train = torch.utils.data.Subset(orig_set, range(n_test, n_total)) 

        print(ds_train[0])
    elif dataset == 'fake':
        # random noise fake data
        ds_train = datasets.FakeData(size = 100000, image_size = (3, resize, resize), num_classes=10, transform = transform)      
        ds_test= datasets.FakeData(size = 100000, image_size = (3, resize, resize), num_classes=10, transform = transform) 
    else:
        print("Dataset name unknown!")

    if percentage < 1.0:
        # Get full dataset targets
        targets = ds_train.targets
        num_classes = 10
        total_samples = len(ds_train)
        fraction = 1 / 5
        samples_per_class = int((total_samples * fraction) / num_classes)

        # Map class label to list of indices
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_to_indices[int(label)].append(idx)

        # Sample equal number from each class
        selected_indices = []
        for label in range(num_classes):
            indices = class_to_indices[label]
            sampled = random.sample(indices, samples_per_class)
            selected_indices.extend(sampled)

        # Create subset dataset
        subset_ds_train = torch.utils.data.Subset(ds_train, selected_indices)
        ds_train = subset_ds_train
        
    return ds_train, ds_test

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


class SpriteDataset(Dataset):
    """Sprite dataset class"""
    def __init__(self, root, transform, target_transform):
        self.images = np.load(os.path.join(root, "sprites_1788_16x16.npy"))
        self.labels = np.load(os.path.join(root, "sprite_labels_nc_1788_16x16.npy"))
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.target_transform(self.labels[idx])
        return image, label

    def __len__(self):
        return len(self.images)


def maybe_download(filename, url, dir, force=False):
    """Download a file if not present, and make sure it's the right size."""
    file_dir = f"{dir}/{filename}"
    if force or not os.path.exists(file_dir):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, file_dir, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(file_dir)
    return filename

def resize_x_dataset(x_dataset, size, channel_to_rgb=False):
    if channel_to_rgb:
        transform = T.Compose([T.Resize([size, size]),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            # gray -> GRB 3 channel (lambda function)
            T.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0])]) 
    else:
        transform = T.Compose([T.Resize([size, size]),
                            T.Normalize(mean=[0.0, 0.0, 0.0],
                                        std=[1.0, 1.0, 1.0])])
    new_x = []
    for i in range(len(x_dataset)):
        new_x.append(transform(x_dataset[i].clone().detach()))
    stacked_x = torch.stack(new_x, dim=0)
    return stacked_x

def to_torch_ds(ds, transform):
    torch_ds = []
    for i in range(len(ds)):
        img = ds[i]['image'].convert("RGB")
        # torch_ds.append([transform(ds[i]['pixel_values']), ds[i]['label']])
        torch_ds.append([transform(img), ds[i]['label']])
    return torch_ds

def create_fl_trust_root_dataset(fl_train, fltrust_root_iid, noniid_type, non_iid_param, fl_class_num, sample_size):
    train_targets = [x[1] for x in fl_train]
    if fltrust_root_iid:
        indices = random.sample(list(range(len(fl_train))), sample_size)    
    else:
        if noniid_type == "p-degree":
            # p_degree non iid, same p_degree as FL clients
            p_degree = non_iid_param
            selected_class = random.choice(list(range(fl_class_num)))
            indices = []
            
            for class_id in range(fl_class_num):
                idxs = [idx for idx, target in enumerate(train_targets) if target == class_id]
                if class_id == selected_class:
                    sub_sample_size = math.floor(sample_size * p_degree)
                else:
                    sub_sample_size = math.floor(sample_size * (1-p_degree)/(fl_class_num-1))
                indices += random.sample(idxs, sub_sample_size)
        elif noniid_type == 'label_skew':
            # print("fltrust noniid label skew")
            indices = []
            # print("non_iid_param", non_iid_param)
            num_selected = int(non_iid_param * fl_class_num) # noniid_param = 1.0, select all classes
            selected_classes = random.sample(list(range(fl_class_num)), k = num_selected) # all classes
            sub_sample_size = int(sample_size/num_selected)
            # print("sub_sample_size", sub_sample_size)
            # print("selected_classes", selected_classes)
            for label in selected_classes:
                label_indices = [idx for idx, target in enumerate(train_targets) if target == label]
                idxs = random.sample(label_indices, sub_sample_size)
                indices += idxs
    ds = torch.utils.data.Subset(fl_train, indices)

    return ds


def quantity_based_label_dist_skew(fl_class_num, client_num, label_skew_alpha):
    # label_skew_alpha
    # class labels per client
    class_per_client = math.floor(fl_class_num * label_skew_alpha)
    labels_per_client_dict = {k: [] for k in range(client_num)} 

    # old code number of each label is not same for the whole clients set
    # count = 0
    # max_guesses_allowed = 500
    # while set(np.concatenate(list(labels_per_client_dict.values()))) != set(list(range(fl_class_num))) or count < max_guesses_allowed:
    #     for client_id in range(client_num):
    #         labels_per_client_dict[client_id] = np.sort(np.random.choice(fl_class_num, class_per_client, replace=False))
    #     count+=1
    
    # label_counts = Counter(np.concatenate(list(labels_per_client_dict.values())))


    count = 0
    max_guesses_allowed = 500000
    divergence= 10
    # divergence>=3
    while  (divergence >=3 or \
            set(np.concatenate(list(labels_per_client_dict.values()))) != set(list(range(fl_class_num))) ) and \
            count < max_guesses_allowed:
        
        for client_id in range(client_num):
            labels_per_client_dict[client_id] = np.sort(np.random.choice(fl_class_num, class_per_client, replace=False))
        
        label_counts = Counter(np.concatenate(list(labels_per_client_dict.values())))
        divergence = max(label_counts.values()) - min(label_counts.values())      
        # print(divergence)
        count+=1
    print("quantity_based_label_dist_skew: count", count)
    label_counts = Counter(np.concatenate(list(labels_per_client_dict.values())))
    return labels_per_client_dict, label_counts


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.samples = []
        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            self.samples.append((img, label))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    
def load_or_cache_imagenette(split='train', transform=None, cache_dir='./data/cache'):
    os.makedirs(cache_dir, exist_ok=True)

    pt_file = os.path.join(cache_dir, f'imagenette_{split}.pt')
    image_root = f'./data/imagenette2-320/{split}'

    if os.path.exists(pt_file):
        print(f"[INFO] Loading cached Imagenette {split} dataset from {pt_file}")
        data_list = torch.load(pt_file)
        return ListDataset(data_list)

    print(f"[INFO] Caching Imagenette {split} dataset to {pt_file} (Only done once)")
    base_dataset = datasets.ImageFolder(root=image_root, transform=transform)

    # Preload all images into memory
    data_list = [base_dataset[i] for i in range(len(base_dataset))]

    # Save it
    torch.save(data_list, pt_file)

    return ListDataset(data_list)