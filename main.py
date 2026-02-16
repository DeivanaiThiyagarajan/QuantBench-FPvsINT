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

def train_model(model, trainloader, valloader, epochs, results_dir):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for batch_X, batch_y in trainloader:
            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)
            
            # Forward pass
            outputs = model(batch_X)
                
            # Convert one-hot to class indices if needed
            if batch_y.dim() > 1 and batch_y.size(1) > 1:
                loss = criterion(outputs, torch.argmax(batch_y, dim=1))
                target = torch.argmax(batch_y, dim=1)
            else:
                loss = criterion(outputs, batch_y.squeeze())
                target = batch_y.squeeze()
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = total_loss / len(trainloader)
        accuracy = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_X, val_y in valloader:
                val_X = val_X.to(model.device)
                val_y = val_y.to(model.device)
                
                val_outputs = model(val_X)
                
                if val_y.dim() > 1 and val_y.size(1) > 1:
                    v_loss = criterion(val_outputs, torch.argmax(val_y, dim=1))
                    v_target = torch.argmax(val_y, dim=1)
                else:
                    v_loss = criterion(val_outputs, val_y.squeeze())
                    v_target = val_y.squeeze()
                    
                val_loss += v_loss.item()
                _, v_predicted = torch.max(val_outputs.data, 1)
                val_total += v_target.size(0)
                val_correct += (v_predicted == v_target).sum().item()

        val_avg_loss = val_loss / len(valloader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}, "
                  f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if val_avg_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
    return avg_loss, accuracy

def evaluate_model(model, testloader, results_dir):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for test_X, test_y in testloader:
            test_X = test_X.to(model.device)
            test_y = test_y.to(model.device)
            
            test_outputs = model(test_X)
            
            if test_y.dim() > 1 and test_y.size(1) > 1:
                t_loss = criterion(test_outputs, torch.argmax(test_y, dim=1))
                t_target = torch.argmax(test_y, dim=1)
            else:
                t_loss = criterion(test_outputs, test_y.squeeze())
                t_target = test_y.squeeze()
                
            test_loss += t_loss.item()
            _, t_predicted = torch.max(test_outputs.data, 1)
            test_total += t_target.size(0)
            test_correct += (t_predicted == t_target).sum().item()
    test_avg_loss = test_loss / len(testloader)
    test_accuracy = test_correct / test_total   
    print(f"Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return test_avg_loss, test_accuracy

def log_weight_statistics(weights_biases, log_file):
    stats = {}
    for name, values in weights_biases.items():
        values_array = np.array(values)
        stats[name] = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array))
        }
    
    with open(log_file, 'w') as f:
        json.dump(stats, f, indent=4)

def quantize_weights(weights_biases, quantized_file, num_bits = 8, symmetric = False, per_channel = False):
    qmin = -(2**(num_bits-1))
    qmax = (2**(num_bits-1)) - 1

    quantized_dict = {}
    for name, arr in weights_biases.items():
        w = np.array(arr, dtype=np.float32)

        # ---------- PER CHANNEL ----------
        if per_channel and w.ndim > 1:
            scales = []
            q_data = []

            for i in range(w.shape[0]):
                w_channel = w[i]
                w_min = w_channel.min()
                w_max = w_channel.max()
                if symmetric:
                    max_val = max(abs(w_min), abs(w_max))
                    scale = max_val / qmax if max_val > 0 else 1.0
                else:
                    scale = (w_max - w_min) / (qmax - qmin) if w_max != w_min else 1.0

                q = np.round(w_channel / scale).astype(np.int32)
                q = np.clip(q, qmin, qmax)
                scales.append(float(scale))
                q_data.append(q.tolist())

            quantized_dict[name] = {
                'scales': scales,
                'quantized': q_data,
                "zero_point": 0,
                "per_channel": True,
                "num_bits": num_bits
            }
        else:
            if symmetric:
                max_val = max(abs(w.min()), abs(w.max()))
                scale = max_val / qmax if max_val > 0 else 1.0
            else:
                w_min = w.min()
                w_max = w.max()
                scale = (w_max - w_min) / (qmax - qmin) if w_max != w_min else 1.0

            q = np.round(w / scale).astype(np.int32)
            q = np.clip(q, qmin, qmax)
            quantized_dict[name] = {
                'scale': float(scale),
                'quantized': q.tolist(),
                "zero_point": 0,
                "per_channel": False,
                "num_bits": num_bits
            }
    with open(quantized_file, "w") as f:
        json.dump(quantized_dict, f, indent=4)

    print(f"Quantized weights saved to {quantized_file}")


def quantize_input(input_data, num_bits=8, symmetric=False):
    """
    Quantize input data to fixed-point representation.
    
    Returns:
        int_input: quantized input as integer array
        input_scale: scale factor for dequantization
        input_zero_point: zero point for asymmetric quantization
    """
    qmin = -(2**(num_bits-1))
    qmax = (2**(num_bits-1)) - 1
    
    input_array = np.array(input_data, dtype=np.float32)
    
    if symmetric:
        max_val = max(abs(input_array.min()), abs(input_array.max()))
        scale = max_val / qmax if max_val > 0 else 1.0
        zero_point = 0
    else:
        min_val = input_array.min()
        max_val = input_array.max()
        scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        zero_point = int(np.round(-min_val / scale))
    
    int_input = np.round(input_array / scale + zero_point).astype(np.int32)
    int_input = np.clip(int_input, qmin, qmax)
    
    return int_input, float(scale), int(zero_point)


def compute_output_scale_per_layer(input_scale, weight_scale, output_scale=None):
    """
    Compute rescale factor to prevent scale accumulation.
    
    Formula: rescale_factor = (input_scale * weight_scale) / output_scale
    
    If output_scale not provided, use the product (assumes unquantized output ref)
    """
    if output_scale is None:
        output_scale = input_scale * weight_scale
    
    rescale_factor = (input_scale * weight_scale) / output_scale if output_scale != 0 else 1.0
    return rescale_factor


def dequantize(int_data, scale, zero_point=0):
    """Convert quantized integer back to floating point."""
    return (int_data - zero_point) * scale

def quantized_model(model):
    for name

def evaluate_quantized_model(model, testloader, results_dir, num_bits=8, symmetric=False):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for test_X, test_y in testloader:
            test_X = test_X.cpu().numpy()
            q_input, input_scale, input_zero_point = quantize_input(test_X, num_bits=num_bits, symmetric=symmetric)
            q_input_tensor = torch.from_numpy(q_input).to(model.device)
            test_y = test_y.to(model.device)






def main(args):
    set_seed(args.seed)
    results_dir = ensure_results_dir()

    train_set, _ = DatasetSelector.get_dataset(args.dataset, train=True, data_dir=args.data_dir)
    input_shape = train_set[0][0].shape

    logger.info(f"Loading dataset: {args.dataset}")
    trainloader, valloader, testloader, num_classes = DatasetSelector.get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        data_dir=args.data_dir
    )

    # Load model and dataset
    logger.info(f"Loading model: {args.model}")
    model = ModelSelector.get_model(args.model, input_shape, num_classes)

    # Train and evaluate the model (placeholder)
    logger.info("Starting training and evaluation...")

    train_model(model, trainloader, valloader, args.epochs, results_dir)

    # Evaluate on test set (placeholder)
    logger.info("Evaluating on test set...")

    evaluate_model(model, testloader, results_dir)

    #extract weights and bias learned by the model and save them in a json file
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()   

    with open(os.path.join(results_dir, f"{args.model}_{args.dataset}_weights.json"), 'w') as f:
        json.dump(weights, f, indent=4)

    log_weight_statistics(weights, os.path.join(results_dir, f"{args.model}_{args.dataset}_weight_stats.json"))

    quantize_weights(weights, os.path.join(results_dir, f"{args.model}_{args.dataset}_quantized_weights.json"), num_bits=args.num_bits, symmetric=args.symmetric, per_channel=args.per_channel)

    evaluate_quantized_model(model, testloader, results_dir, num_bits=args.num_bits, symmetric=args.symmetric)