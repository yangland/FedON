import sys
import os
import logging
from datetime import datetime
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models import *
from helper import *
import random
from collections import Counter, OrderedDict
import seaborn as sns
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import networkx as nx
import math
from functools import partial
import itertools


# Set up logging and output directory
def setup_output_directory(dataset, model_name, dir_path='experiments'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_dir = f"{dir_path}/{timestamp}_{dataset}_{model_name}"
    os.makedirs(folder_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(folder_dir, 'experiment.log')),
            logging.StreamHandler()
        ]
    )
    return folder_dir

# Configuration class to hold all parameters
class Config:
    def __init__(self):
        self.iters = 20
        self.learning_rate = 0.01 # 0.001 for smaller datasets
        self.attack = "untargeted"
        self.model_name = "ResNet18" # "ResNet18", "VGG13", "MobileNetV2", "SqueezeNet", "ConvNet",
        self.num_classes = 10
        self.mali_client_num = 10
        self.dataset = "cifar10"    # "mnist", "fmnist", "cifar10", "SVHN", "imagenette"
        self.device = "cuda:3" if torch.cuda.is_available() else "cpu"
        self.client_num = 100
        self.noniid_type = "dirichlet"
        self.dirichlet_alpha = 100
        self.batch_size = 32

# Visualization functions
class Visualizer:
    @staticmethod
    def plot_l2_norms(l2_norms, output_path):
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(l2_norms) + 1), l2_norms, color='skyblue')
        plt.xlabel("Iteration")
        plt.ylabel("L2 Norm of Model Update")
        plt.title("L2 Norms per Iteration")
        plt.xticks(range(1, len(l2_norms) + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'l2_norms.png'))
        plt.close()
        logging.info("Saved L2 norms plot")

    @staticmethod
    def plot_accuracy(accuracy_list, output_path):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, color='orange', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Iteration")
        plt.xticks(range(1, len(accuracy_list) + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'accuracy.png'))
        plt.close()
        logging.info("Saved accuracy plot")

    @staticmethod
    def plot_heatmap(matrix, title, filename, output_path, cmap="viridis"):
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=False, cmap=cmap, square=True)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Iteration")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, filename))
        plt.close()
        logging.info(f"Saved heatmap: {title}")

    @staticmethod
    def plot_trajectory(l2_norms, flattened_updates, output_path):
        angles = [0.0]
        for i in range(1, len(flattened_updates)):
            u_prev = flattened_updates[i - 1]
            u_curr = flattened_updates[i]
            cos_sim = F.cosine_similarity(u_prev, u_curr, dim=0).item()
            angle_diff = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            sign = torch.sign(torch.dot(u_prev, u_curr))
            if sign < 0:
                angle_diff = -angle_diff
            angles.append(angle_diff)

        absolute_angles = [0.0]
        for i in range(1, len(angles)):
            absolute_angles.append(absolute_angles[-1] + angles[i])

        x, y = [0], [0]
        for i in range(len(l2_norms)):
            dx = l2_norms[i] * np.cos(absolute_angles[i])
            dy = l2_norms[i] * np.sin(absolute_angles[i])
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)

        plt.figure(figsize=(6, 5))
        for i in range(len(l2_norms)):
            plt.arrow(x[i], y[i], x[i+1]-x[i], y[i+1]-y[i],
                      head_width=0.05*max(l2_norms), length_includes_head=True,
                      color='royalblue', alpha=0.8)

        plt.title("Model Update Trajectory (Chronological Arrows)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'trajectory.png'))
        plt.close()
        logging.info("Saved trajectory plot")

    @staticmethod
    def plot_attack_accuracy(l2_distances, attack_accuracy_list, max_attack_l2_dist, output_path):
        plt.figure(figsize=(12, 6))
        plt.plot(l2_distances[:20], attack_accuracy_list[:20], marker='o', linestyle='-', color='b', linewidth=2)
        plt.title('Attack Accuracy vs. $L_2$', fontsize=14)
        plt.xlabel('L2 distance', fontsize=12)
        plt.ylabel('Attack Accuracy (ACC)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.axvline(x=max_attack_l2_dist, color='r', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'attack_accuracy.png'))
        plt.close()
        logging.info("Saved attack accuracy plot")

    @staticmethod
    def plot_triangle_relationship(l2_diff_matrix, output_path):
        model_names = ['original', 'benign', 'attacked']
        model_types = {'original': 'original', 'benign': 'benign', 'attacked': 'attacked'}
        color_map = {'original': 'yellow', 'benign': 'green', 'attacked': 'red'}

        a = l2_diff_matrix[0, 1].item()
        b = l2_diff_matrix[0, 2].item()
        c = l2_diff_matrix[1, 2].item()

        cos_CAB = (a**2 + b**2 - c**2) / (2 * a * b)
        angle_CAB = math.acos(cos_CAB)

        pos = {
            'original': (0, 0),
            'benign': (a, 0),
            'attacked': (b * math.cos(angle_CAB), b * math.sin(angle_CAB))
        }

        G = nx.Graph()
        for name in model_names:
            G.add_node(name)

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                G.add_edge(model_names[i], model_names[j], weight=round(l2_diff_matrix[i, j].item(), 2))

        node_colors = [color_map[model_types[name]] for name in model_names]
        plt.figure(figsize=(6, 5))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, edgecolors='black', linewidths=1.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', alpha=0.8)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
        )

        plt.title("Exact L2 Distance Triangle Between Models", pad=20, fontsize=13)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'triangle_relationship.png'))
        plt.close()
        logging.info("Saved triangle relationship plot")

    @staticmethod
    def plot_sensitivity_lines(results_matrix, layer_names, L2_dists, results_list_all, dataset, model_name, output_path):
        plt.figure(figsize=(10, 6))
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', '*']
        colors = plt.cm.tab20(np.linspace(0, 1, len(layer_names)))
        style_combinations = list(itertools.product(line_styles, markers))

        max_accuracy = results_matrix.max() * 100
        threshold = 0.75 * max_accuracy

        # Compute min accuracy per layer
        min_accs = [curve.min() * 100 for curve in results_matrix]
        layer_infos = list(zip(layer_names, results_matrix, min_accs, colors))

        # Filter layers with min acc below threshold
        filtered_layers = [info for info in layer_infos if info[2] < threshold]

        # If too many, keep only 10 with lowest min acc
        if len(filtered_layers) > 10:
            filtered_layers = sorted(filtered_layers, key=lambda x: x[2])[:10]

        for idx, (layer_name, acc_curve, _, color) in enumerate(filtered_layers):
            acc_curve = acc_curve * 100
            line_style, marker = style_combinations[idx % len(style_combinations)]

            plt.plot(
                L2_dists, acc_curve, label=layer_name,
                linestyle=line_style, marker=marker, color=color,
                linewidth=2, markersize=5, alpha=1.0
            )

        if results_list_all is not None:
            plt.plot(
                L2_dists, results_list_all * 100, label="all layers",
                color='black', linestyle='-', linewidth=3,
                marker='X', markersize=7
            )

        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        plt.legend(
            unique.values(), 
            unique.keys(), 
            loc='center left', 
            bbox_to_anchor=(1.04, 0.5), 
            fontsize=9,
            frameon=False
        )

        plt.xlabel('$L_2$ Budget', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Layer Sensitivity vs. ALL Layers Attack ({dataset}, {model_name})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(output_path, 'sensitivity_lines.png'))
        plt.close()
        logging.info("Saved sensitivity lines plot")


# Helper functions
def flatten_update(update_dict, device='cpu'):
    return torch.cat([v.view(-1).to(device) for v in update_dict.values()])

def clone_model_params(model):
    return {name: param.detach().clone() for name, param in model.state_dict().items()}

def compute_model_update(current_state, previous_state):
    return {name: (current_state[name] - previous_state[name]).detach() for name in current_state}

def flatten_model(state_dict):
    return torch.cat([param.flatten() for param in state_dict.values()])

class NegativeCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(NegativeCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        return -self.ce_loss(outputs, labels)

def train_one_epoch_with_negative_ce(model, train_loader, device, max_retries=50):
    def contains_nan(tensor_list):
        return any(param is not None and torch.isnan(param).any() for param in tensor_list)

    model.train()
    neg_ce_loss = NegativeCrossEntropyLoss()
    initial_lr = 0.001
    current_lr = initial_lr
    retry_count = 0

    # Move model state to CPU to reduce GPU memory usage
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    while retry_count < max_retries:
        retry_count += 1

        # Restore model state to GPU
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
        optimizer = optim.SGD(model.parameters(), lr=current_lr)
        nan_detected = False

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = neg_ce_loss(outputs, labels)
            loss.backward()

            if contains_nan([p.grad for p in model.parameters()]):
                nan_detected = True
                break

            optimizer.step()

            if contains_nan([p.data for p in model.parameters()]):
                nan_detected = True
                break

        if nan_detected:
            current_lr *= 0.1
            print(f"[Retry {retry_count}] NaN detected. Reducing lr to {current_lr} and restarting epoch.")
        else:
            break

    if retry_count == max_retries:
        print("Maximum retries reached. Proceeding with last attempted state.")

    # Compute parameter differences
    G_m = {}
    final_state = model.state_dict()
    for name in original_state:
        final_tensor = final_state[name].cpu()
        if torch.isnan(final_tensor).any():
            print(f"Warning: NaN in parameter {name} after training. Using zero change.")
            G_m[name] = torch.zeros_like(original_state[name])
        else:
            G_m[name] = final_tensor - original_state[name]

    # Restore original model state to GPU
    model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
    return G_m, current_lr




def apply_attack_magnitude(model, G_m, average_l2_norm, magnitudes):
    attacked_models = []
    l2_distances = []
    original_params = deepcopy(flatten_model(model.state_dict()))

    for magnitude in magnitudes:
        l2_budget = magnitude * average_l2_norm
        print(f"Applying attack with L2 budget: {l2_budget}")

        malicious_model = apply_attack_to_all_layers(model, G_m, l2_budget)

        attacked_models.append(malicious_model)
        malicious_params = flatten_model(malicious_model.state_dict())
        distance = torch.norm(original_params - malicious_params, p=2)
        l2_distances.append(distance.item())

    return attacked_models, l2_distances



def apply_attack_to_specific_layer(model, G_m, l2_budget, layer_name):
    new_model = deepcopy(model)
    
    # Extract the gradient/direction for the specified layer
    direction = G_m[layer_name]
    
    # Compute the L2 norm of the direction
    direction_norm = direction.norm()
    
    # If norm is non-zero, scale to match l2_budget
    if direction_norm != 0:
        scaled_direction = direction * (l2_budget / direction_norm)
    else:
        scaled_direction = torch.zeros_like(direction)
    
    # Apply the scaled perturbation to the specified layer only
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            if name == layer_name:
                param.data += scaled_direction.to(param.device)

    return new_model


def apply_attack_to_all_layers(model, G_m, l2_budget):
    # Only consider named_parameters for norm calculation
    total_norm = torch.sqrt(sum(
        (G_m[name].float().norm() ** 2)
        for name, param in model.named_parameters()
        if name in G_m
    ))

    if total_norm == 0:
        return model

    scale = l2_budget / total_norm
    logging.info(f"Attack scale: {scale}")

    attacked_model = deepcopy(model)
    with torch.no_grad():
        for name, param in attacked_model.named_parameters():
            if name in G_m:
                param.data += scale * G_m[name].float().to(param.device)

    return attacked_model



def evaluate_and_record_per_layer_sensitivities(model, G_m, max_l2, num_records, test_data, device):
    G_m = {name: g_m.float() for name, g_m in G_m.items()}
    layer_names = [name for name, _ in model.named_parameters()]
    l2_dists = np.linspace(0, max_l2, num_records)
    results_matrix = np.zeros((len(layer_names), len(l2_dists)))
    
    logging.info(f"Evaluating sensitivity for layers: {layer_names}")
    
    for layer_index, layer_name in enumerate(layer_names):
        logging.info(f"Processing layer: {layer_name}")
        for idx, l2_budget in enumerate(l2_dists):
            attacked_model = apply_attack_to_specific_layer(model, G_m, l2_budget, layer_name)
            test_loss, accuracy, per_class_accuracy, attack_success_rate = fl_validation(
                attacked_model, nn.CrossEntropyLoss(), test_data, server_batch_size=32, class_num=10,
                attack_method='none', mali_labels=[0, 7], device=device
            )
            results_matrix[layer_index, idx] = accuracy
            logging.info(f"Layer {layer_name}, L2 {l2_budget:.2f}: Accuracy {accuracy:.4f}")

    return results_matrix, layer_names, l2_dists

def evaluate_attack_on_all_layers(model, G_m, max_l2, num_records, test_data, device):
    l2_dists = np.linspace(0, max_l2, num_records)
    results_list = []
    G_m = {k: v.float() for k, v in G_m.items()}
    
    logging.info("Evaluating attack on all layers")
    
    for l2_budget in l2_dists:
        logging.info(f"Processing L2 budget: {l2_budget:.2f}")
        attacked_model = apply_attack_to_all_layers(model, G_m, l2_budget)
        test_loss, accuracy, per_class_accuracy, attack_success_rate = fl_validation(
            attacked_model, nn.CrossEntropyLoss(), test_data, server_batch_size=32, class_num=10,
            attack_method='none', mali_labels=[0, 7], device=device
        )
        results_list.append(accuracy)
        logging.info(f"L2 {l2_budget:.2f}: Accuracy {accuracy:.4f}")

    return np.array(results_list), l2_dists


def get_dataset(dataset_name):
    # Get datasets
    transform_img = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    transform_img2 = T.Compose([
        T.Resize((224, 224)),  # This will resize the image to 320x320
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "mnist":
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform_img)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform_img)
    elif dataset_name == "fmnist":
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_img)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_img)
    elif dataset_name == "cifar10":
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_img)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_img)
    elif dataset_name == "SVHN":
        train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform_img)
        test_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform_img)
    elif dataset_name == "imagenette":
        train_data = datasets.Imagenette(root='./data', split='train', size="320px", download=False, transform=transform_img2)
        test_data = datasets.Imagenette(root='./data', split='val', size="320px", download=False, transform=transform_img2)
    return train_data, test_data

def calculate_distance_matrix(flattened_updates, device='cpu'):
    """
    Calculates cosine similarity and L2 distance matrices between flattened model updates.
    Optimized to run on CPU to reduce GPU memory usage.
    """
    num_updates = len(flattened_updates)
    cos_diff_matrix = torch.zeros(num_updates, num_updates)
    l2_diff_matrix = torch.zeros(num_updates, num_updates)

    for i in range(num_updates):
        vec_i = flattened_updates[i].to('cpu')
        for j in range(i + 1, num_updates):
            vec_j = flattened_updates[j].to('cpu')

            # Normalize vectors to avoid division by zero
            if vec_i.norm() == 0 or vec_j.norm() == 0:
                cos_sim = 0.0
            else:
                cos_sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()

            l2_diff = torch.norm(vec_i - vec_j).item()

            cos_diff_matrix[i, j] = 1 - cos_sim
            cos_diff_matrix[j, i] = 1 - cos_sim
            l2_diff_matrix[i, j] = l2_diff
            l2_diff_matrix[j, i] = l2_diff

    return cos_diff_matrix, l2_diff_matrix



def get_flattened_updates(model_updates, device='cpu'):
    """
    Flattens and collects model updates in a sorted order.

    Args:
        model_updates (List): Dictionary mapping client indices to model updates.
        device (str): Device to move updates to before flattening. Default is 'cpu'.

    Returns:
        List[torch.Tensor]: List of flattened updates.
    """
    flattened_updates = []
    for i in range(len(model_updates)):
        flat_update = flatten_update(model_updates[i], device=device).detach()
        flattened_updates.append(flat_update)
        del flat_update
        torch.cuda.empty_cache()
    return flattened_updates

# Main experiment function
def main():
    config = Config()
    output_folder = setup_output_directory(config.dataset, config.model_name)
    
    logging.info("Starting experiment")
    logging.info(f"Configuration: {vars(config)}")
    
    train_data, test_data = get_dataset(config.dataset)

    logging.info(f"Loaded {config.dataset} dataset")

    # Create FL clients
    fl_data_dict, labels_per_client_dict, label_count_per_client_dict =\
        create_data_fl_clients(client_num=config.client_num,
                             noniid_type=config.noniid_type,
                             label_skew_alpha =None,
                             p_degree=None,
                             dirichlet_alpha=config.dirichlet_alpha,
                             fl_train=train_data,
                             fl_class_num=config.num_classes)

    # Combine datasets from clients 0 to 9 as malicious clients's pooled data
    combined_dataset = ConcatDataset([fl_data_dict[i] for i in range(config.mali_client_num)])
    train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    
    # Get models
    model_class = get_model(config.model_name)
    model_fn = partial(model_class, num_classes=config.num_classes)
    
    # Initialize models
    model0 = model_fn().to(config.device)
    model0_sd = deepcopy(model0.state_dict())
    optimizer0 = optim.SGD(model0.parameters(), lr=config.learning_rate)
    c_criterion0 = nn.CrossEntropyLoss()

    # Duplicated model for attack
    modelt = model_fn().to(config.device)
    modelt.load_state_dict(model0_sd)
    c_criterion1 = nn.CrossEntropyLoss()

    # Benign training loop
    model_updates = []
    model_state_dicts = []
    prev_model_state = clone_model_params(model0)
    accuracy_list = []
    logging.info("Starting training loop")
    
    for iter in range(config.iters):
        # Train for 1 epoch
        train_c_model(train_loader, model0, c_criterion0, optimizer0, attack_method='none', client_id=0,
                      malicious_client_ids=[], optimizer_poison=None, dba_poisoning_per_batch=0.2, device=config.device)

        # Compute and save the model update
        current_model_state = clone_model_params(model0)
        model_updates.append(compute_model_update(current_model_state, prev_model_state))
        model_state_dicts.append(clone_model_params(model0))
        prev_model_state = current_model_state

        # Validation accuracy
        test_loss, accuracy, per_class_accuracy, attack_success_rate = fl_validation(
            model0, c_criterion0, test_data, server_batch_size=32, class_num=10, 
            attack_method='none', mali_labels=[0,7], device=config.device
        )
        accuracy_list.append(accuracy)
        logging.info(f"Iteration {iter}: Accuracy = {accuracy:.4f}")

    # Compute flattened updates and metrics
    flattened_updates = get_flattened_updates(model_updates, device='cpu')
    flattened_state_dicts = get_flattened_updates(model_state_dicts, device='cpu')

    l2_norms = [u.norm().item() for u in flattened_updates]
    last_l2_norm = l2_norms[-1]
    logging.info(f"The last model update's L2 norm: {last_l2_norm:.4f}")

    # Compute similarity matrices
    update_cos_diff_matrix, update_l2_diff_matrix = calculate_distance_matrix(flattened_updates, config.device)
    sd_cos_diff_matrix, sd_l2_diff_matrix = calculate_distance_matrix(flattened_state_dicts, config.device)

    # Generate visualizations
    Visualizer.plot_l2_norms(l2_norms, output_folder)
    Visualizer.plot_accuracy(accuracy_list, output_folder)
    Visualizer.plot_heatmap(update_cos_diff_matrix.numpy(), "Model Updates Cosine Difference Matrix (1 - cos similarity)", 
                          "update_cosine_diff.png", output_folder, "viridis")
    Visualizer.plot_heatmap(update_l2_diff_matrix.numpy(), "Model Updates L2 Distance Matrix", 
                          "update_l2_diff.png", output_folder, "magma")
    Visualizer.plot_heatmap(sd_cos_diff_matrix.numpy(), "Model State Dict Cosine Difference Matrix (1 - cos similarity)", 
                          "sd_cosine_diff.png", output_folder, "viridis")
    Visualizer.plot_heatmap(sd_l2_diff_matrix.numpy(), "Model State Dict L2 Distance Matrix", 
                          "sd_l2_diff.png", output_folder, "magma")    
    Visualizer.plot_trajectory(l2_norms, flattened_updates, output_folder)

    # Attack experiments
    logging.info("Starting attack experiments")
    modelt.load_state_dict(model0.state_dict())
    G_m, final_lr = train_one_epoch_with_negative_ce(modelt, train_loader, config.device, max_retries=50)
    magnitudes = (
    np.linspace(0, 0.01, 6, endpoint=True).tolist() +
    np.linspace(0.01, 0.1, 6, endpoint=True).tolist() +
    np.linspace(0.1 + 1e-6, 1, 6, endpoint=True).tolist() +
    np.linspace(1, 20, 20, endpoint=True).tolist()
)
    
    modelt.load_state_dict(model0.state_dict())
    attacked_models, l2_distances = apply_attack_magnitude(modelt, G_m, last_l2_norm, magnitudes)
    
    attack_accuracy_list = []
    for i, attacked_model in enumerate(attacked_models):
        test_loss, accuracy, per_class_accuracy, attack_success_rate = fl_validation(
            attacked_model, c_criterion1, test_data, server_batch_size=32, class_num=10,
            attack_method='none', mali_labels=[0,7], device=config.device
        )
        attack_accuracy_list.append(accuracy*100)
        logging.info(f"Attack {i}: Magnitude {magnitudes[i]}, Accuracy {accuracy*100:.2f}%")

    max_att_index = next(
    (i for i, acc in enumerate(attack_accuracy_list) if acc <= 10),
    min(enumerate(attack_accuracy_list), key=lambda x: x[1])[0]
    )

    logging.info(f"max_att_index: {max_att_index}")
    max_attack_l2_dist = l2_distances[max_att_index]
    logging.info(f"l2_distances[: {l2_distances}")
    logging.info(f"Maximum attack L2 distance: {max_attack_l2_dist:.4f}")

    Visualizer.plot_attack_accuracy(l2_distances, attack_accuracy_list, max_attack_l2_dist, output_folder)

    # Triangle relationship plot
    original_flat = flatten_model(model0_sd)
    benign_flat = flatten_model(modelt.state_dict())
    attacked_flat = flatten_model(attacked_models[max_att_index].state_dict())
    tri_models = [original_flat, benign_flat, attacked_flat]
    
    _, l2_diff_matrix = calculate_distance_matrix(tri_models, config.device)
    
    Visualizer.plot_triangle_relationship(l2_diff_matrix, output_folder)
    
    # === Save model weights to disk ===
    torch.save(model0_sd, os.path.join(output_folder, 'model0.pt'))
    logging.info("Saved initial model weights to model0.pt")

    torch.save(modelt.state_dict(), os.path.join(output_folder, 'modelt.pt'))
    logging.info("Saved benign model weights to modelt.pt")

    torch.save(attacked_models[max_att_index].state_dict(), os.path.join(output_folder, 'model_att.pt'))
    logging.info("Saved attacked model weights to model_att.pt")

    # Sensitivity analysis
    logging.info("Starting sensitivity analysis")
    results_matrix, layer_names, l2_dists = evaluate_and_record_per_layer_sensitivities(
        modelt, G_m, max_attack_l2_dist, 10, test_data, config.device
    )
    results_list_all, l2_dists_all = evaluate_attack_on_all_layers(
        modelt, G_m, max_attack_l2_dist, 10, test_data, config.device
    )
    
    Visualizer.plot_sensitivity_lines(
        results_matrix, layer_names, l2_dists_all, results_list_all, 
        config.dataset, config.model_name, output_folder
    )

    logging.info("Experiment completed successfully")

if __name__ == "__main__":
    main()