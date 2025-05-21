import os
import sys
import argparse
from datetime import datetime
import logging
from functools import partial
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import torchvision.datasets as datasets
import torchvision.transforms as T

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Project specific imports
project_path = os.path.abspath('..')
sys.path.append(project_path)

from utilities import *
from helper import *
from models import *

# ---------------------- CONFIGURATION & LOGGER ---------------------- #

def create_experiment_dir(dataset, model_name, base_dir="experiments"):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f"{timestamp}_{dataset}_{model_name}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ---------------------- PLOTTING ---------------------- #

def save_plot_line(x, y, title, xlabel, ylabel, output_dir, model_name, dataset):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)  # Rotate x-ticks to 45 degrees
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    filename = f"{title.replace(' ', '_')}_{dataset}_{model_name}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches='tight')  # Ensure everything fits
    plt.close()
    return path

def save_plot_heatmap(matrix, title, output_dir, model_name, dataset, xticks=None, yticks=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='viridis', xticklabels=xticks, yticklabels=yticks)
    plt.title(title)
    plt.xlabel("Iteration" if xticks is None else "")
    plt.ylabel("Iteration" if yticks is None else "")
    plt.tight_layout()
    filename = f"{title.replace(' ', '_')}_{dataset}_{model_name}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

# ---------------------- HELPER FUNCTIONS ---------------------- #

def clone_model_params(model):
    return {name: param.detach().clone() for name, param in model.state_dict().items()}

def compute_model_update(current_state, previous_state):
    return {name: (current_state[name] - previous_state[name]).detach() for name in current_state}

def get_l2_norm(update_dict):
    return torch.cat([v.view(-1) for v in update_dict.values()]).norm().item()

def scale_update_to_l2(update_dict, target_norm):
    current_norm = get_l2_norm(update_dict)
    scale = target_norm / (current_norm + 1e-10)
    return {k: (v * scale).detach() for k, v in update_dict.items()}

def scale_layer_update(update_dict, target_layer, target_norm):
    scaled = {}
    for name, param in update_dict.items():
        if name == target_layer:
            norm = param.norm()
            scale = target_norm / (norm + 1e-10)
            scaled[name] = (param * scale).detach()
        else:
            scaled[name] = torch.zeros_like(param)
    return scaled

def apply_update(model, update_dict):
    new_state = model.state_dict()
    for name in update_dict:
        new_state[name] = new_state[name] + update_dict[name]
    model.load_state_dict(new_state)

# ---------------------- MAIN EXPERIMENT ---------------------- #

def run_experiment(args):
    # Setup
    exp_dir = create_experiment_dir(args.dataset, args.model_name)
    logger = setup_logger(os.path.join(exp_dir, "log.txt"))
    logger.info(f"Experiment directory: {exp_dir}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Data transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    # Dataset loading
    if args.dataset == "mnist":
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif args.dataset == "fmnist":
        train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    elif args.dataset == "cifar10":
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

    # Federated simulation
    fl_data_dict, _, _ = create_data_fl_clients(
                           client_num=100,
                           noniid_type="dirichlet",
                           label_skew_alpha =None,
                           p_degree=None,
                           dirichlet_alpha=100,
                           fl_train=train_data,
                           fl_class_num=10)
    combined_dataset = ConcatDataset([fl_data_dict[i] for i in range(10)])
    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    # Model setup
    model_class = get_model(args.model_name)
    model_fn = partial(model_class, num_classes=10)
    model = model_fn().to(device)
    initial_state = clone_model_params(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Step 1: Train benign model
    logger.info("Training benign model...")
    for epoch in range(args.benign_epochs):
        train_c_model(train_loader, model, criterion, optimizer, attack_method='none', client_id=0,
                      malicious_client_ids=[], optimizer_poison=None, dba_poisoning_per_batch=0.2, device=device)

    benign_state = clone_model_params(model)
    benign_update = compute_model_update(initial_state, benign_state)
    benign_norm = get_l2_norm(benign_update)
    logger.info(f"Benign update L2 norm: {benign_norm:.4f}")

    # Step 2: Conduct malicious training (1 epoch)
    logger.info("Generating malicious model via 1 epoch negative entropy training...")
    malicious_model = model_fn().to(device)
    malicious_model.load_state_dict(clone_model_params(model))
    optimizer_mal = optim.SGD(malicious_model.parameters(), lr=args.lr)
    criterion_mal = NegativeEntropyLoss()

    train_c_model(train_loader, malicious_model, criterion_mal, optimizer_mal, attack_method='none', client_id=0,
                  malicious_client_ids=[], optimizer_poison=None, dba_poisoning_per_batch=0.2, device=device)

    # Step 3: Calculate malicious update
    malicious_state = clone_model_params(malicious_model)
    mal_update = compute_model_update(malicious_state, benign_state)
    mal_norm = get_l2_norm(mal_update)
    logger.info(f"Malicious update L2 norm: {mal_norm:.4f}")

    # Step 4: Sweep multipliers to find effective L2 budget range
    logger.info("Sweeping multipliers to find effective L2 budget range...")
    multipliers = np.linspace(0, 30, 20)
    acc_vs_multiplier = []
    l2_norms = []

    for m in multipliers:
        temp_model = model_fn().to(device)
        temp_model.load_state_dict(clone_model_params(model))
        scaled_update = scale_update_to_l2(mal_update, m * mal_norm)
        apply_update(temp_model, scaled_update)

        _, acc, _, _ = fl_validation(temp_model, criterion, test_data, server_batch_size=32, class_num=10,
                                     attack_method='none', mali_labels=[0, 7], device=device)
        acc_vs_multiplier.append(acc)
        l2_norms.append(get_l2_norm(scaled_update))
        logger.info(f"Multiplier: {m:.2f} → Acc: {acc:.4f} → L2: {l2_norms[-1]:.2f}")

    save_plot_line(
        x=l2_norms,
        y=acc_vs_multiplier,
        title="Accuracy vs Scaled Attack L2 Norm",
        xlabel="L2 Norm",
        ylabel="Accuracy",
        output_dir=exp_dir,
        model_name=args.model_name,
        dataset=args.dataset
    )

    # Step 5: Define refined L2 budget list
    min_l2 = 0
    max_att_index = min(enumerate(acc_vs_multiplier), key=lambda x: x[1])[0]
    max_l2 = l2_norms[max_att_index]
    budget_range = np.linspace(min_l2, max_l2, 10)

    logger.info(f"Selected L2 range for detailed analysis: {budget_range[0]:.2f} → {budget_range[-1]:.2f}")

    acc_vs_l2 = []
    for budget in budget_range:
        temp_model = model_fn().to(device)
        temp_model.load_state_dict(clone_model_params(model))
        scaled_update = scale_update_to_l2(mal_update, budget)
        apply_update(temp_model, scaled_update)

        _, acc, _, _ = fl_validation(temp_model, criterion, test_data, server_batch_size=32, class_num=10,
                                     attack_method='none', mali_labels=[0, 7], device=device)
        acc_vs_l2.append(acc)
        logger.info(f"L2 Budget: {budget:.2f} → Accuracy: {acc:.4f}")

    save_plot_line(
        x=budget_range,
        y=acc_vs_l2,
        title="Accuracy vs Refined L2 Budget",
        xlabel="L2 Norm",
        ylabel="Accuracy",
        output_dir=exp_dir,
        model_name=args.model_name,
        dataset=args.dataset
    )

    # Step 6: Per-layer sensitivity
    logger.info(f"Evaluating per-layer sensitivity with L2 budget: {args.l2_budget:.2f}")
    layer_names = list(mal_update.keys())
    layer_accs = []

    for lname in layer_names:
        temp_model = model_fn().to(device)
        temp_model.load_state_dict(clone_model_params(model))
        layer_update = scale_layer_update(mal_update, lname, args.l2_budget)
        apply_update(temp_model, layer_update)

        _, acc, _, _ = fl_validation(temp_model, criterion, test_data, server_batch_size=32, class_num=10,
                                     attack_method='none', mali_labels=[0, 7], device=device)
        layer_accs.append((lname, acc))
        logger.info(f"Layer: {lname} → Accuracy: {acc:.4f}")

    layer_labels = [x[0] for x in layer_accs]
    layer_values = [x[1] for x in layer_accs]
    save_plot_line(
        x=layer_labels,
        y=layer_values,
        title="Per-Layer Attack Impact",
        xlabel="Layer",
        ylabel="Accuracy",
        output_dir=exp_dir,
        model_name=args.model_name,
        dataset=args.dataset
    )

    logger.info("Experiment complete.")

# ---------------------- MAIN ---------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Attack Budget and Sensitivity Experiment")

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'fmnist', 'cifar10'])
    parser.add_argument('--model_name', type=str, default='ResNet18', choices=['ResNet18', 'ConvNet', 'VGG13'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--benign_epochs', type=int, default=5)
    parser.add_argument('--l2_budget', type=float, default=20.0)

    args = parser.parse_args()
    run_experiment(args)
