import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
import numpy as np
import os
import json
from datetime import datetime
from model_selector import ModelSelector
from dataset_selector import DatasetSelector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_results_dir():
    if not os.path.exists('results'):
        os.makedirs('results')
    return 'results'

def main(args):
    set_seed(args.seed)
    results_dir = ensure_results_dir()

    logger.info(f"Loading dataset: {args.dataset}")
    trainloader, testloader, num_classes = DatasetSelector.get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        data_dir=args.data_dir
    )

    # Load model and dataset
    logger.info(f"Loading model: {args.model}")
    model_cls, input_size = ModelSelector.get_model(args.model)
    model = model_cls(num_classes=DatasetSelector.AVAILABLE_DATASETS[args.dataset][1])
