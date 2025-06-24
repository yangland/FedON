import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy
from functools import partial
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os, sys

# === Import project-specific modules ===
project_path = os.path.abspath('..')  # Modify if needed
sys.path.append(project_path)
from models import *
from utilities import *
from helper import *

# ------------------------
# Data & Model Utilities
# ------------------------

def get_model_and_data(model_name, data_name, num_classes, percent, device):
    model_class = get_model(model_name)
    model_fn = partial(model_class, num_classes=num_classes)
    model = model_fn().to(device)
    transform = transforms.ToTensor()
    if data_name == "mnist":
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif data_name == "cifar10":
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    train_loader = get_partial_loader(train_data, percent=percent, batch_size=64)
    return model, train_loader, test_loader

def get_partial_loader(dataset, percent=0.2, batch_size=32, shuffle=True, num_workers=1):
    total_len = len(dataset)
    subset_len = int(total_len * percent)
    indices = random.sample(range(total_len), subset_len)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def flatten_update(update_dict):
    return torch.cat([update_dict[k].view(-1) for k in sorted(update_dict.keys()) if update_dict[k].dtype == torch.float32])

def normalize_vec(vec):
    return vec / (vec.norm() + 1e-10)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ------------------------
# Malicious Update Generator
# ------------------------

def get_attack_direction(model, train_loader, device, initial_lr=0.001, num_epochs=2, max_retries=10):
    def contains_nan(tensor_list):
        return any(param is not None and torch.isnan(param).any() for param in tensor_list)

    model = deepcopy(model).to(device)
    model.train()
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    retry_count = 0
    current_lr = initial_lr
    optimizer = optim.SGD(model.parameters(), lr=current_lr)

    while retry_count < max_retries:
        retry_count += 1
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
        nan_detected = False

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels)
                loss.backward()
                if contains_nan([p.grad for p in model.parameters()]):
                    nan_detected = True
                    break
                optimizer.step()
                if contains_nan([p.data for p in model.parameters()]):
                    nan_detected = True
                    break
            if nan_detected:
                break

        if nan_detected:
            current_lr *= 0.1
            optimizer = optim.SGD(model.parameters(), lr=current_lr)
            print(f"[Retry {retry_count}] NaN detected. Reducing lr to {current_lr}")
        else:
            break

    final_state = model.state_dict()
    grads = {}
    for name in original_state:
        grads[name] = final_state[name].cpu() - original_state[name]
    return grads

# ------------------------
# Plot: Dim-wise % Change Comparison
# ------------------------

def plot_dimwise_change_histogram(
    benign_update_dict_convnet,
    malicious_update_dict_convnet,
    benign_update_dict_resnet,
    malicious_update_dict_resnet,
    bins=100,
    filename="dimwise_change_comparison.pdf"
):
    def compute_dimwise_percent_change(benign_update, malicious_update):
        percent_list = []

        # Only use keys that are common AND float32
        valid_keys = sorted(
            k for k in benign_update.keys()
            if k in malicious_update and
            benign_update[k].dtype == torch.float32 and
            malicious_update[k].dtype == torch.float32 and
            benign_update[k].shape == malicious_update[k].shape
        )

        flat_b = torch.cat([benign_update[k].contiguous().view(-1) for k in valid_keys])
        flat_m = torch.cat([malicious_update[k].contiguous().view(-1) for k in valid_keys])

        diff = torch.abs(flat_b - flat_m)
        total = diff.sum().item() + 1e-8
        percent_per_dim = (diff / total * 100.0).cpu().numpy()
        return percent_per_dim


    convnet_percent = compute_dimwise_percent_change(benign_update_dict_convnet, malicious_update_dict_convnet)
    resnet_percent = compute_dimwise_percent_change(benign_update_dict_resnet, malicious_update_dict_resnet)

    # # Plotting
    # plt.figure(figsize=(7, 3.5))
    # sns.histplot(convnet_percent, bins=bins, color="#0072B2", label="ConvNet", stat="density", kde=False, alpha=0.7)
    # sns.histplot(resnet_percent, bins=bins, color="#D55E00", label="ResNet18", stat="density", kde=False, alpha=0.7)

    # plt.xlabel("Per-Dimension % of Total Update Difference")
    # plt.ylabel("Density")
    # plt.title("Dimension-wise Update Change Comparison")
    # plt.legend()

    # # --- Zoom in on meaningful x-range ---
    # max_val = max(np.percentile(convnet_percent, 99.9), np.percentile(resnet_percent, 99.9))
    # plt.xlim(0, max_val * 1.1)  # Add 10% padding

    # plt.tight_layout()
    # plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    # print(f"✅ Saved dimension-wise comparison plot: {filename}")
    # plt.show()
    
    K = 20
    top_convnet = np.sort(convnet_percent)[-K:][::-1]
    top_resnet = np.sort(resnet_percent)[-K:][::-1]

    plt.figure(figsize=(6.5, 3.5))
    plt.plot(top_convnet, label="ConvNet", marker='o')
    plt.plot(top_resnet, label="ResNet18", marker='x')
    plt.xticks(range(1, K+1))
    plt.xlabel(f"Top-K (K={K}) Dimensions")
    plt.ylabel("% of Total Update Difference")
    # plt.title("Top-K Dimension Contributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

# ------------------------
# Run One Model and Collect Updates
# ------------------------

def run_model_update(model_name, data_name, device, data_percent=0.2, lr=0.001):
    model0, train_loader, test_loader = get_model_and_data(
        model_name=model_name,
        data_name=data_name,
        num_classes=10,
        percent=data_percent,
        device=device
    )

    model = deepcopy(model0).to(device)

    # Benign training (1 round)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    # Benign update
    benign_update = {}
    for k, v in model.state_dict().items():
        benign_update[k] = v.cpu() - model0.state_dict()[k].cpu()

    # Evaluate benign accuracy
    acc_benign = evaluate(model, test_loader, device)

    # Get malicious gradient direction
    grads = get_attack_direction(model0, train_loader, device)

    # Apply scaled malicious update
    acc_mal = 1.0
    scale_factor = 1.0
    while acc_mal > 0.15 and scale_factor < 100:
        poisoned_model = deepcopy(model0).to(device)
        poisoned_state = deepcopy(model0.state_dict())

        for name in poisoned_state:
            if name in grads:
                grad = grads[name].to(device) * scale_factor
                if poisoned_state[name].dtype != torch.float32:
                    poisoned_state[name] = poisoned_state[name].float()
                poisoned_state[name] += grad

        poisoned_model.load_state_dict(poisoned_state)
        acc_mal = evaluate(poisoned_model, test_loader, device)

        if acc_mal > 0.10:
            scale_factor *= 1.5

    # Final malicious update
    malicious_update_scaled = {k: v * scale_factor for k, v in grads.items()}

    # Optional logging
    print(f"[{model_name}] Benign Acc: {acc_benign:.4f}, Malicious Acc: {acc_mal:.4f}")

    return benign_update, malicious_update_scaled


# ------------------------
# Main Execution
# ------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_name = "cifar10"

    # Run for ConvNet
    print("\n=== Running ConvNet ===")
    benign_conv, mal_conv = run_model_update("ConvNet", data_name, device)

    # Run for ResNet18
    print("\n=== Running ResNet18 ===")
    benign_resnet, mal_resnet = run_model_update("ResNet18", data_name, device)

    # Plot comparison
    plot_dimwise_change_histogram(
        benign_update_dict_convnet=benign_conv,
        malicious_update_dict_convnet=mal_conv,
        benign_update_dict_resnet=benign_resnet,
        malicious_update_dict_resnet=mal_resnet,
        bins=100,
        filename="dimwise_change_convnet_vs_resnet18.pdf"
    )

if __name__ == "__main__":
    main()
