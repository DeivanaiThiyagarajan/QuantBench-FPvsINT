import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

from dataset_selector import DatasetSelector
from model_selector import ModelSelector

class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int,
                 dropout: float = 0.0, use_batch_norm: bool = False):
        super(SimpleMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)


class SimpleConvNet(nn.Module):
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(SimpleConvNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    # Example usage
    dataset_name = 'CIFAR10'

    train_set, num_classes = DatasetSelector.get_dataset(dataset_name, train=True)
    input_shape = train_set[0][0].shape

    #logger.info(f"Loading dataset: {args.dataset}")
    trainloader, valloader, testloader, num_classes = DatasetSelector.get_dataloaders(
        dataset_name=dataset_name,
        batch_size=32
    )
    # Load model and dataset
    #logger.info(f"Loading model: {args.model}")
    model = ModelSelector.get_model('SimpleMLP', input_shape, num_classes)

    print(model)
