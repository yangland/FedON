# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from copy import deepcopy
# import matplotlib.pyplot as plt
# import numpy as np
import sys
import os

# Manually specify the path to your project folder
project_path = os.path.abspath('..')  # Or provide the full path if needed
sys.path.append(project_path)

# Now you can import from the sibling folder
from utilities import *
from helper import *

import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models import *
import random
from collections import Counter
from collections import OrderedDict
import seaborn as sns
from copy import deepcopy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from torch.utils.data import Subset
import random
from matplotlib.ticker import PercentFormatter

# ------------------------
# Utility Functions
# ------------------------

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    elif data_name == "fmnist":
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)       
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
    return torch.cat([v.view(-1) for v in update_dict.values()])

def normalize_vec(vec):
    return vec / (vec.norm() + 1e-10)

def cosine_distance(u, v):
    return 1 - F.cosine_similarity(u, v, dim=0).item()


def plot_param_change_histogram(benign_update, malicious_update, model_name="Model", save=True):
    """
    Plots a histogram showing the percentage of total L2 norm difference per named parameter.
    """
    param_names = []
    percent_diffs = []
    
    total_norm_diff = 0.0
    norm_diffs = {}

    for name in benign_update:
        if benign_update[name].dtype != torch.float32:
            continue  # Skip non-float params
        diff = (malicious_update[name] - benign_update[name]).view(-1)
        norm = torch.norm(diff).item()
        norm_diffs[name] = norm
        total_norm_diff += norm

    for name, norm in norm_diffs.items():
        param_names.append(name)
        percent_diffs.append(100.0 * norm / (total_norm_diff + 1e-10))  # Avoid div-by-zero

    plt.figure(figsize=(6, 3))
    plt.barh(param_names, percent_diffs, color="#D55E00")
    plt.xlabel("Percentage of Total L2 Difference (%)")
    plt.title(f"{model_name}: Layer-wise Update Difference")
    plt.tight_layout()
    
    if save:
        filename = f"param_diff_percent_{model_name}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"✅ Saved update difference histogram: {filename}")
    plt.show()


def plot_sign_mismatch_histogram(benign_update, malicious_update, model_name="Model", save=True):
    """
    Plots a histogram showing sign mismatch percentage between benign and malicious updates per parameter tensor.
    """
    param_names = []
    mismatch_rates = []

    for name in benign_update:
        if benign_update[name].dtype != torch.float32:
            continue
        b = benign_update[name].view(-1)
        m = malicious_update[name].view(-1)

        signs_b = torch.sign(b)
        signs_m = torch.sign(m)
        mismatch = (signs_b != signs_m).float()
        mismatch_rate = mismatch.mean().item() * 100.0

        param_names.append(name)
        mismatch_rates.append(mismatch_rate)

    plt.figure(figsize=(6, 3))
    plt.barh(param_names, mismatch_rates, color="#0072B2")
    plt.xlabel("Sign Mismatch Rate (%)")
    plt.title(f"{model_name}: Sign Mismatch Between Updates")
    plt.tight_layout()

    if save:
        filename = f"sign_mismatch_{model_name}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"✅ Saved sign mismatch histogram: {filename}")
    plt.show()


# Ensure consistent ordering by sorting keys
def flatten_update(update_dict):
    return torch.cat([update_dict[k].view(-1) for k in sorted(update_dict.keys())])

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
                loss = -F.cross_entropy(outputs, labels)  # Gradient ascent (maximize loss)
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
    g_m = flatten_update(grads)
    g_m_unit = normalize_vec(g_m)
    return grads, g_m_unit


def plot_dimwise_change_histogram(
    benign_update_dict_convnet,
    malicious_update_dict_convnet,
    benign_update_dict_resnet,
    malicious_update_dict_resnet,
    bins=100,
    save=True,
    filename="dimwise_change_comparison.pdf"
):
    """
    Plots side-by-side histograms of % contribution of each dimension's change
    between benign and malicious updates for ConvNet and ResNet18.
    """
    def compute_dimwise_percent_change(benign_update, malicious_update):
        # Flatten and compute absolute difference
        flat_b = torch.cat([v.view(-1) for k, v in sorted(benign_update.items()) if v.dtype == torch.float32])
        flat_m = torch.cat([v.view(-1) for k, v in sorted(malicious_update.items()) if v.dtype == torch.float32])
        diff = torch.abs(flat_b - flat_m)
        total = diff.sum().item() + 1e-8
        percent_per_dim = (diff / total * 100.0).cpu().numpy()
        return percent_per_dim

    # Compute per-dimension % change
    convnet_percent = compute_dimwise_percent_change(benign_update_dict_convnet, malicious_update_dict_convnet)
    resnet_percent = compute_dimwise_percent_change(benign_update_dict_resnet, malicious_update_dict_resnet)

    # Plotting
    plt.figure(figsize=(6, 3))
    sns.histplot(convnet_percent, bins=bins, color="#0072B2", label="ConvNet", stat="density", kde=False, alpha=0.7)
    sns.histplot(resnet_percent, bins=bins, color="#D55E00", label="ResNet18", stat="density", kde=False, alpha=0.7)

    plt.xlabel("Per-Dimension % of Total Update Difference")
    plt.ylabel("Density")
    plt.title("Distribution of Dimension-wise Update Changes")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"✅ Saved dimension-wise comparison plot: {filename}")
    plt.show()


def plot_all_metrics(
    benign_l2_list,
    malicious_l2_list,
    benign_acc_list,
    malicious_acc_list,
    cosine_dist_list,
    data_name="Dataset",
    model_name="Model",
    colors=None
):
    """
    Plots L2 Norms, Accuracies, and Cosine Distance in one figure with 3 subplots.
    
    Args:
        benign_l2_list: List of L2 norms for benign updates.
        malicious_l2_list: List of L2 norms for malicious updates.
        benign_acc_list: List of accuracies for benign updates (0 to 1).
        malicious_acc_list: List of accuracies for malicious updates (0 to 1).
        cosine_dist_list: List of cosine distances between benign and malicious updates.
        data_name: Name of the dataset.
        model_name: Name of the model.
        colors: Dict specifying colors for 'benign', 'malicious', and 'cos'.
    """
    # Default colors if not provided
    if colors is None:
        colors = {
            "benign": "blue",
            "malicious": "red",
            "cos": "purple"
        }
    
    iterations = list(range(len(benign_l2_list)))

    # Create 3 vertically stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(3.25, 2.8), sharex=True, dpi=300
    )

    # Font and line settings
    label_fontsize = 8
    tick_fontsize = 7
    legend_fontsize = 7
    line_width = 1.2

    # --- Plot 1: L2 Norms ---
    ax1.plot(iterations, benign_l2_list, color=colors["benign"], linewidth=line_width)
    ax1.plot(iterations, malicious_l2_list, color=colors["malicious"], linewidth=line_width)
    ax1.set_ylabel("L2 Norm", fontsize=label_fontsize)
    ax1.legend(["Benign", "Malicious"], loc='upper left', fontsize=legend_fontsize, frameon=False)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.tick_params(labelsize=tick_fontsize)

    # --- Plot 2: Accuracy ---
    ax2.plot(iterations, benign_acc_list, color=colors["benign"], linewidth=line_width)
    ax2.plot(iterations, malicious_acc_list, color=colors["malicious"], linewidth=line_width)
    ax2.set_ylabel("Accuracy", fontsize=label_fontsize)
    ax2.legend(["Benign", "Malicious"], loc='upper left', fontsize=legend_fontsize, frameon=False)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.tick_params(labelsize=tick_fontsize)

    # --- Plot 3: Cosine Distance ---
    ax3.plot(iterations, cosine_dist_list, color=colors["cos"], label="Cosine Dist.", linewidth=line_width)
    ax3.set_xlabel("Iteration", fontsize=label_fontsize)
    ax3.set_ylabel("Cos. Dist.", fontsize=label_fontsize)
    ax3.legend(loc='upper left', fontsize=legend_fontsize, frameon=False)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.tick_params(labelsize=tick_fontsize)

    # --- Layout and Saving ---
    plt.tight_layout(pad=0.8)
    filename = f"plot_all_metrics_{data_name}_{model_name}.pdf"
    plt.savefig(filename, bbox_inches='tight', dpi=300, format="pdf")
    plt.show()
    print(f"✅ Saved combined plot: {filename}")
    
# ------------------------
# Evaluation
# ------------------------

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
# Main Logic
# ------------------------

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data_name="cifar10"
    model_name="ResNet18"
    lr = 0.001
    iters = 10
    data_precent = 0.2
    # --- Setup --- # <-- Replace with your model + dataloader code
    model0, train_loader, test_loader = get_model_and_data(
        model_name=model_name,
        data_name=data_name,
        num_classes=10,
        percent=data_precent,
        device=device
    )
    

    benign_l2_list = []
    malicious_l2_list = []
    cosine_dist_list = []
    benign_acc_list = []
    malicious_acc_list = []

    model = deepcopy(model0).to(device)

    for iteration in range(iters):
        print(f"\n=== Iteration {iteration+1} ===")

        # Step 1: Train benign model for 1 round
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        # Step 2: Get benign update
        benign_update = {}
        for k, v in model.state_dict().items():
            benign_update[k] = v.cpu() - model0.state_dict()[k].cpu()
        benign_flat = flatten_update(benign_update)
        benign_unit = normalize_vec(benign_flat)
        benign_l2 = benign_flat.norm().item()

        # Step 2.5: Evaluate benign model accuracy
        acc_benign = evaluate(model, test_loader, device)

        # Step 3: Find malicious update that reduces accuracy to ~10%
        print("Searching for malicious update...")
        acc_mal = 1.0
        scale_factor = 1.0
        grads, g_m_unit = get_attack_direction(model0, train_loader, device)
        

        while acc_mal > 0.15 and scale_factor < 100:
            # Apply malicious update to a copy of model0
            poisoned_model = deepcopy(model0).to(device)
            poisoned_state = deepcopy(model0.state_dict())

            for name in poisoned_state:
                if name in grads:
                    grad = grads[name].to(device) * scale_factor
                    # Ensure matching data type before addition
                    if poisoned_state[name].dtype != torch.float32:
                        poisoned_state[name] = poisoned_state[name].float()
                    poisoned_state[name] += grad  # Apply malicious gradient

            poisoned_model.load_state_dict(poisoned_state)
            acc_mal = evaluate(poisoned_model, test_loader, device)

            if acc_mal > 0.10:
                scale_factor *= 1.5  # Increase poison strength iteratively

        # After loop: flatten malicious update for analysis
        g_m_scaled_dict = {k: v * scale_factor for k, v in grads.items()}



        g_m_scaled = flatten_update(g_m_scaled_dict)
        g_m_l2 = g_m_scaled.norm().item()
        g_m_unit_scaled = normalize_vec(g_m_scaled)
        print("benign_unit shape:", benign_unit.shape)
        print("g_m_unit_scaled shape:", g_m_unit_scaled.shape)
        cos_dist = cosine_distance(benign_unit, g_m_unit_scaled)


        print(f"Benign L2: {benign_l2:.4f}, Malicious L2: {g_m_l2:.4f}, "
        f"Cosine Distance: {cos_dist:.4f}, "
        f"Benign Acc: {acc_benign:.4f}, Malicious Acc: {acc_mal:.4f}")


        # Record
        benign_l2_list.append(benign_l2)
        malicious_l2_list.append(g_m_l2)
        cosine_dist_list.append(cos_dist)
        benign_acc_list.append(acc_benign)
        malicious_acc_list.append(acc_mal)


        # Update model0 for next iteration
        model0.load_state_dict(model.state_dict())

    # -------------------------
    # Plot X - L2 Norms and Accuracies
    # -------------------------
    colors = {
        "benign": "#0072B2",   # Blue
        "malicious": "#D55E00",# Vermillion (orange-red)
        "cos": "#999999"       # Gray
    }

    plot_param_change_histogram(
        benign_update,
        g_m_scaled_dict,
        model_name=model_name
    )

    plot_sign_mismatch_histogram(
        benign_update,
        g_m_scaled_dict,
        model_name=model_name
    )


    plot_all_metrics(
        benign_l2_list,
        malicious_l2_list,
        benign_acc_list,
        malicious_acc_list,
        cosine_dist_list,
        data_name=data_name,
        model_name=model_name,
        colors=colors
    )

if __name__ == "__main__":
    main()
