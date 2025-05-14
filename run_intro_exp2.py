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
from torch.amp import autocast
from torch.amp import GradScaler
import gc

# Set the environment variable before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
        self.iters = 25
        self.learning_rate = 0.01 # 0.001 for smaller datasets
        self.attack = "untargeted"
        self.model_name = "ConvNet" # "ResNet18", "VGG13", "MobileNetV2", "SqueezeNet", "ConvNet", "EfficientNetB0"
        self.num_classes = 10
        self.mali_client_num = 2
        self.dataset = "fmnist"    # "mnist", "fmnist", "cifar10", "SVHN", "Imagenette"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.client_num = 10
        self.noniid_type = "dirichlet"
        self.dirichlet_alpha = 1.0
        self.batch_size = 32 # 32

# Visualization functions
class Visualizer:
    @staticmethod
    def plot_l2_norms(l2_norms, output_path, file_name="l2_norms"):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(l2_norms) + 1), l2_norms, color='purple', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("L2 Norm")
        plt.title("L2 Norms per Iteration")
        plt.xticks(range(1, len(l2_norms) + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{file_name}.png'))
        plt.close()
        logging.info(f"Saved {file_name} L2 norms plot")

    @staticmethod
    def plot_cos_dist(cos_distances, output_path, file_name="cos_distances"):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(cos_distances) + 1), cos_distances, color='red', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Cosine Distance")
        plt.title("Cosine Distance per Iteration")
        plt.xticks(range(1, len(cos_distances) + 1))
        
        # Add horizontal line at y=1 (expected value if perfectly opposite directions)
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{file_name}.png'))
        plt.close()
        logging.info(f"Saved {file_name} cosine distances plot")

    @staticmethod
    def plot_comparative_l2_norms(server_l2_distances, attack_l2_distances, server_to_initial_distances, output_path):
        plt.figure(figsize=(10, 5))
        
        # Plot server model L2 update norms
        plt.plot(range(1, len(server_l2_distances) + 1), server_l2_distances, color='skyblue', marker='o', label='Server L2 Norms')
        
        # Plot attack L2 norms
        plt.plot(range(1, len(attack_l2_distances) + 1), attack_l2_distances, color='orange', marker='x', label='Attack L2 Norms')
        
        # Plot server to initial L2 distance norms
        # plt.plot(range(1, len(server_to_initial_distances) + 1), server_to_initial_distances, color='green', marker='s', label='Server to Initial')

        plt.xlabel("Iteration")
        plt.ylabel("$L_2$ Norm")
        plt.title("Comparative $L_2$ Norms per Iteration")
        plt.xticks(range(1, len(server_l2_distances) + 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'comparative_l2_norms.png'))
        plt.close()
        logging.info("Saved comparative L2 norms plot")


    @staticmethod
    def plot_comparative_accuracy(benign_accuracy, malicious_accuracy, output_path):
        plt.figure(figsize=(10, 5))
        
        plt.plot(range(1, len(benign_accuracy) + 1), benign_accuracy, color='blue', marker='o', label='Benign Accuracy')
        plt.plot(range(1, len(malicious_accuracy) + 1), malicious_accuracy, color='red', marker='x', label='Malicious Accuracy')

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Benign and Malicious Accuracy per Iteration")
        plt.xticks(range(1, len(benign_accuracy) + 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'comparative_accuracy.png'))
        plt.close()
        logging.info("Saved comparative accuracy plot")

    @staticmethod
    def plot_accuracy(accuracy_list, output_path, file_name="accuracy"):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, color='orange', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Iteration")
        plt.xticks(range(1, len(accuracy_list) + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{file_name}.png'))
        plt.close()
        logging.info(f"Saved {file_name} accuracy plot")

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


def apply_attack_with_l2_budgets(model, G_m, l2_budgets):
    attacked_models = []
    l2_distances = []
    original_params = deepcopy(flatten_model(model.state_dict()))

    for l2_budget in l2_budgets:
        logging.info(f"Applying attack with absolute L2 budget: {l2_budget}")
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
    # logging.info(f"Attack scale: {scale}")

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
    elif dataset_name == "Imagenette":
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


def calculate_cosine_similarity_matrix(vectors):
    num_vectors = len(vectors)
    cos_sim_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(num_vectors):
            if i != j:
                cos_sim_matrix[i, j] = F.cosine_similarity(vectors[i], vectors[j], dim=0).item()
            else:
                cos_sim_matrix[i, j] = 1.0  # Cosine similarity with itself is 1

    return cos_sim_matrix


def calculate_cos_dist_matrix(vectors):
    num_vectors = len(vectors)
    cos_dist_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(num_vectors):
            if i != j:
                cos_dist_matrix[i, j] = 1 - F.cosine_similarity(vectors[i], vectors[j], dim=0).item()
            else:
                cos_dist_matrix[i, j] = 0.0  # Cosine distance with itself is 0

    return cos_dist_matrix


# Modified process_clients_in_batches function
def process_clients_in_batches(model_fn, server_model_sd, benign_clients, fl_data_dict, batch_size, batch_size_clients, config):
    scaler = GradScaler(device=config.device)
    model_weights = []
    
    # Pre-allocate a single model for all clients to reuse
    base_model = model_fn().to(config.device)
    base_model.load_state_dict(server_model_sd)
    
    for batch_start in range(0, len(benign_clients), batch_size_clients):
        batch_clients = benign_clients[batch_start:batch_start + batch_size_clients]
        
        for client_id in batch_clients:
            logging.info(f"Training client {client_id}...")
            
            # Create fresh optimizer for each client
            optimizer = torch.optim.SGD(base_model.parameters(), lr=config.learning_rate)
            
            client_train_loader = torch.utils.data.DataLoader(
                fl_data_dict[client_id],
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            # Train client
            base_model.train()
            for inputs, labels in client_train_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                
                with autocast(device_type=config.device):
                    outputs = base_model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # Get client update and reset model
            with torch.no_grad():
                client_weights = {}
                for name, param in base_model.state_dict().items():
                    client_weights[name] = param.detach().cpu()
                model_weights.append(client_weights)
            
            # Clear gradients and optimizer state
            del optimizer
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reset model to server state for next client
            base_model.load_state_dict(server_model_sd)
    
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return model_weights


def get_dynamic_l2_budgets(previous_best_l2=None, model_update_l2_norm=None):
    if previous_best_l2 is None and model_update_l2_norm is not None:
        # Initial iteration - use model_update_l2_norm as reference
        base_l2 = model_update_l2_norm
        logger.info(f"Initial model update L2 norm with magnitude: {base_l2}")
        return (
            np.linspace(0.01 * base_l2, 0.1 * base_l2, 6, endpoint=False).tolist() +
            np.linspace(0.1 * base_l2, 1 * base_l2, 6, endpoint=False).tolist() +
            np.linspace(1 * base_l2, 20 * base_l2, 9, endpoint=True).tolist()
        )
    elif previous_best_l2 is not None:
        # Subsequent iterations - use previous best L2 as reference
        min_l2 = max(0.001, previous_best_l2 * 0.5)
        max_l2 = previous_best_l2 * 2
        logger.info(f"Previous best L2 norm: {previous_best_l2}")
        
        # Extended scaling for very large values
        if previous_best_l2 < 0.01:
            min_l2 = max(0.0001, min_l2)
            max_l2 = min(0.02, max_l2)
        elif previous_best_l2 < 0.1:
            min_l2 = max(0.001, min_l2)
            max_l2 = min(0.2, max_l2)
        elif previous_best_l2 < 1:
            min_l2 = max(0.01, min_l2)
            max_l2 = min(2, max_l2)
        elif previous_best_l2 < 10:
            min_l2 = max(0.1, min_l2)
            max_l2 = min(20, max_l2)
        elif previous_best_l2 < 100:
            min_l2 = max(1, min_l2)
            max_l2 = min(200, max_l2)
        elif previous_best_l2 < 1000:
            min_l2 = max(10, min_l2)
            max_l2 = min(2000, max_l2)
        else:  # For values >= 1000
            min_l2 = max(100, min_l2)
            max_l2 = previous_best_l2 * 1.5
        
        return np.linspace(min_l2, max_l2, 10).tolist()
    else:
        # Fallback (shouldn't normally happen)
        logger.info(f"Fallback: using default L2 budget range")
        return np.linspace(0.01, 20, 15).tolist()


# In the main loop, replace the magnitudes definition with:
previous_best_l2 = None  # Initialize before the loop

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
                             label_skew_alpha=None,
                             p_degree=None,
                             dirichlet_alpha=config.dirichlet_alpha,
                             fl_train=train_data,
                             fl_class_num=config.num_classes)

    # Combine datasets from malicious clients to form a pooled dataset
    combined_dataset = torch.utils.data.ConcatDataset([fl_data_dict[i] for i in range(config.mali_client_num)])
    malicious_train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Get models
    model_class = get_model(config.model_name)
    model_fn = partial(model_class, num_classes=config.num_classes, dataset=config.dataset)

    # Initialize server model
    server_model = model_fn().to(config.device)
    # server_model_sd = deepcopy(server_model.state_dict())
    initial_server_flat = flatten_model(server_model.state_dict())
    c_criterion = nn.CrossEntropyLoss()

    # Initialize malicious client model
    malicious_model = model_fn().to(config.device)

    # Benign and malicious client setup
    benign_clients = list(range(config.mali_client_num, config.client_num))
    malicious_clients = list(range(config.mali_client_num))

    accuracy_list = []
    malicious_accuracy_list = []
    server_l2_distances = []  # Track L2 norm of server model updates
    attack_l2_distances = []  # Track L2 norm of best attack
    server_to_initial_distances = []  # Track L2 norm from initial model
    g_m_directions = []  # Track G_m directions per iteration
    cos_distances_gm_neg_update = []  # Track cosine distances between G_m and -model_update

    # Previous server model state
    prev_server_state = clone_model_params(server_model)

    previous_best_l2 = None  # Initialize before the loop

    logging.info("Starting federated training loop")

    for iter in range(config.iters):
        # get the server model state
        server_model_sd = deepcopy(server_model.state_dict())
        
        # Process clients in batches to manage GPU memory usage
        model_weights = process_clients_in_batches(
            model_fn, server_model_sd, benign_clients, fl_data_dict, config.batch_size, 5, config
        )

        # FedAvg: Aggregate model updates and update server_model_sd
        for key in server_model_sd.keys():
            stacked_tensors = [w[key].float() for w in model_weights]  # Ensure tensors are in float
            server_model_sd[key] = torch.mean(torch.stack(stacked_tensors), dim=0)
        server_model.load_state_dict(server_model_sd)

        # Evaluate server model
        with torch.no_grad():  # Ensure no gradients are stored during evaluation
            test_loss, server_accuracy, _, _ = fl_validation(
                server_model, c_criterion, test_data, server_batch_size=32, class_num=10, 
                attack_method='none', mali_labels=[0, 7], device=config.device
            )
        accuracy_list.append(server_accuracy)
        logging.info(f"Iteration {iter}: Server Model Accuracy = {server_accuracy:.4f}")

        # Compute L2 norm between current and previous server model
        current_server_flat = flatten_model(server_model.state_dict())
        prev_server_flat = flatten_model(prev_server_state)
        model_update = current_server_flat - prev_server_flat
        model_update_l2_norm = torch.norm(model_update).item()
        server_l2_distances.append(model_update_l2_norm)

        # Compute L2 distance from initial model
        server_to_initial_distance = torch.norm(current_server_flat - initial_server_flat).item()
        server_to_initial_distances.append(server_to_initial_distance)

        # Load the server model state into the malicious model
        malicious_model.load_state_dict(server_model.state_dict())

        # Malicious client updates using negative cross-entropy
        G_m, final_lr = train_one_epoch_with_negative_ce(malicious_model, malicious_train_loader, config.device)
        
        # Flatten and store the direction of G_m for cosine similarity computation
        g_m_directions.append(flatten_update(G_m, device=config.device))

        # Flatten and store the direction of G_m for cosine similarity computation
        g_m_flat = flatten_update(G_m, device=config.device)
        g_m_directions.append(g_m_flat)

        # Compute cosine similarity between G_m and -model_update
        cos_distance = 1 - torch.nn.functional.cosine_similarity(
            g_m_flat.unsqueeze(0), 
            (-model_update).unsqueeze(0), 
            dim=1
        ).item()
        
        cos_distances_gm_neg_update.append(cos_distance)

        # Get dynamic L2 budgets
        l2_budgets = get_dynamic_l2_budgets(previous_best_l2, model_update_l2_norm)
        logging.info(f"L2 budgets for iteration {iter}: {l2_budgets}")
    
        # magnitudes = (
        #     np.linspace(0.01, 0.1, 6, endpoint=False).tolist() +
        #     np.linspace(0.1, 1, 6, endpoint=False).tolist() +
        #     np.linspace(1, 20, 9, endpoint=True).tolist()
        # )
        
        #logging.info(f"Dynamic magnitudes for iteration {iter}: {magnitudes}")

        # attacked_models, l2_distances_iter = apply_attack_magnitude(malicious_model, G_m, model_update_l2_norm, magnitudes)
        
        attacked_models, l2_distances_iter = apply_attack_with_l2_budgets(
                                    malicious_model, G_m, l2_budgets
                                    )

        # Existing logic to identify the worst attack
        attack_accuracy_list = []
        for attacked_model in attacked_models:
            with torch.no_grad():
                test_loss, attack_accuracy, _, _ = fl_validation(
                    attacked_model, c_criterion, test_data, server_batch_size=32, class_num=10,
                    attack_method='none', mali_labels=[0, 7], device=config.device
                )
            attack_accuracy_list.append(attack_accuracy)

        # Determine the attack with the worst performance
        max_att_index = next(
            (i for i, acc in enumerate(attack_accuracy_list) if acc <= 0.1),  # Adjust threshold as needed
            min(enumerate(attack_accuracy_list), key=lambda x: x[1])[0]
        )
        
        best_attack_model = attacked_models[max_att_index]
        best_attack_accuracy = attack_accuracy_list[max_att_index]
        best_attack_l2_distance = l2_distances_iter[max_att_index]
        
        malicious_accuracy_list.append(best_attack_accuracy)
        attack_l2_distances.append(best_attack_l2_distance)

        logging.info(f"Iteration {iter}: Best Malicious Model Accuracy = {best_attack_accuracy:.4f}, Attack L2 Distance = {best_attack_l2_distance:.4f}, Model Update L2 Norm = {model_update_l2_norm:.4f}")

        # Update the previous server model state
        prev_server_state = clone_model_params(server_model)
        
        # Update previous_best_distance for next iteration
        previous_best_l2 = best_attack_l2_distance

    # Visualizations
    # Visualizer.plot_accuracy(accuracy_list, output_folder, file_name="benign_accuracy")
    # Visualizer.plot_accuracy(malicious_accuracy_list, output_folder, file_name="malicious_accuracy")
    Visualizer.plot_comparative_l2_norms(server_l2_distances, attack_l2_distances, server_to_initial_distances, output_folder)
    Visualizer.plot_l2_norms(server_to_initial_distances, output_path=output_folder, file_name="server_to_initial_l2_norms")
    Visualizer.plot_comparative_accuracy(accuracy_list, malicious_accuracy_list, output_folder)

    # Compute and plot cosine similarity matrix
    Visualizer.plot_cos_dist(cos_distances_gm_neg_update, output_folder, file_name="cos_dist_attack_direction_vs_inverse_of_benign_model_update")
    cos_sim_matrix = calculate_cosine_similarity_matrix(g_m_directions)
    Visualizer.plot_heatmap(cos_sim_matrix, "Cosine Similarity Matrix of G_m Directions", "cosine_similarity_matrix.png", output_folder)

    logging.info("Experiment completed successfully")

if __name__ == "__main__":
    main()