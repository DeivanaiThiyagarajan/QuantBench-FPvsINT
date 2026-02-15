import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetSelector:

    AVAILABLE_DATASETS = {
        'CIFAR10': (torchvision.datasets.CIFAR10, 10),
        'CIFAR100': (torchvision.datasets.CIFAR100, 100),
        'MNIST': (torchvision.datasets.MNIST, 10),
        'SVHN': (torchvision.datasets.SVHN, 10),
        'ImageNet': (torchvision.datasets.ImageNet, 1000),
    }

    @staticmethod
    def list_datasets() -> list:
        return list(DatasetSelector.AVAILABLE_DATASETS.keys())

    @staticmethod
    def get_transforms(dataset_name: str, is_train: bool = True) -> transforms.Compose:
        if dataset_name in ['CIFAR10', 'CIFAR100']:
            return transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                ),
            ])
        
        elif dataset_name == 'MNIST':
            return transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                ),
            ])
        
        elif dataset_name == 'SVHN':
            return transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4376, 0.4437, 0.4728),
                    std=(0.1980, 0.2010, 0.1970)
                ),
            ])
        
        elif dataset_name == 'ImageNet':
            return transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def get_dataset(dataset_name: str, train: bool = True, 
                   num_samples: Optional[int] = None,
                   data_dir: str = './data') -> Tuple[Dataset, int]:

        if dataset_name not in DatasetSelector.AVAILABLE_DATASETS:
            available = ', '.join(DatasetSelector.list_datasets())
            raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {available}")
        
        dataset_class, num_classes = DatasetSelector.AVAILABLE_DATASETS[dataset_name]
        transform = DatasetSelector.get_transforms(dataset_name, is_train=train)
        
        # Create data directory if it doesn't exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dataset with appropriate parameters
        if dataset_name == 'SVHN':
            dataset = dataset_class(
                root=data_dir,
                split='train' if train else 'test',
                download=True,
                transform=transform
            )
        elif dataset_name == 'ImageNet':
            split = 'train' if train else 'val'
            dataset = dataset_class(
                root=data_dir,
                split=split,
                transform=transform
            )
        else:
            dataset = dataset_class(
                root=data_dir,
                train=train,
                download=True,
                transform=transform
            )
        
        # Limit number of samples if requested
        if num_samples is not None and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = Subset(dataset, indices)
        
        split_type = 'train' if train else 'test'
        logger.info(f"Loaded {dataset_name} {split_type} set with {len(dataset)} samples")
        
        return dataset, num_classes

    @staticmethod
    def get_dataloaders(dataset_name: str, batch_size: int = 32,
                       num_train_samples: Optional[int] = None,
                       num_test_samples: Optional[int] = None,
                       num_workers: int = 2,
                       data_dir: str = './data') -> Tuple[DataLoader, DataLoader, int]:

        trainset, num_classes = DatasetSelector.get_dataset(
            dataset_name, train=True, num_samples=num_train_samples, data_dir=data_dir
        )
        
        testset, _ = DatasetSelector.get_dataset(
            dataset_name, train=False, num_samples=num_test_samples, data_dir=data_dir
        )
        
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created dataloaders for {dataset_name}")
        logger.info(f"  Train: {len(trainloader)} batches of size {batch_size}")
        logger.info(f"  Test: {len(testloader)} batches of size {batch_size}")
        logger.info(f"  Classes: {num_classes}")
        
        return trainloader, testloader, num_classes


def main():
    # Example usage
    dataset_name = 'CIFAR10'

    train_set, num_classes = DatasetSelector.get_dataset(dataset_name, train=True)
    print(f"Loaded {dataset_name} train set with {len(train_set)} samples and {num_classes} classes")
    print(f"Example data shape: {train_set[0][0].shape}, label: {train_set[0][1]}")


if __name__ == "__main__":
    main()
