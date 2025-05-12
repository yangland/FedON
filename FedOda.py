import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader
from run_intro_exp import (
    get_dataset,
    train_one_epoch_with_negative_ce,
    apply_attack_to_all_layers,
    calculate_distance_matrix,
    flatten_model,
    clone_model_params
)

class FedOda:
    def __init__(self, model_fn, device, learning_rate=0.01, scaling_factor=8, reduce_factor=0.5):
        # Model-related variables
        self.model_fn = model_fn
        self.device = device
        self.learning_rate = learning_rate
        self.reduce_factor = reduce_factor
        
        # State variables to be saved
        self.model0_sd = None      # Initial model state dict
        self.modelt_sd = None      # Pretrained model state dict
        self.att_direction = None            # Attack direction
        self.last_update_norm = 0  # Last update norm during pretraining
        self.benign_direction = None  # Benign direction
        self.scaling_factor = scaling_factor # Current scaling factor
        
        self.server_model = None  # Server model
        self.previous_server_model = None
        
    def pretrain(self, iters, pooled_data, test_data):
        """
        Pretrain the model and initialize attack parameters
        """
        # Step 1: Initialize models
        model0 = self.model_fn().to(self.device)
        self.model0_sd = deepcopy(model0.state_dict())

        # Train model to convergence
        optimizer_t = optim.SGD(model0.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Initialize variables for tracking updates
        prev_params = None
        update_norms = []  # List to store all update norms
        pooled_data_loader = DataLoader(pooled_data, batch_size=32, shuffle=True, num_workers=0)
        vali_accuracies = []
        
        for iteration in range(iters):
            for inputs, labels in pooled_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer_t.zero_grad()
                outputs = model0(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_t.step()

            # validate the model on the validation set
            test_loss, accuracy, per_class_accuracy = \
                model_validation(model=model0, 
                                 criterion=criterion, 
                                 test_data=test_data, 
                                 device=self.device, 
                                 class_num=10)
            vali_accuracies.append(accuracy)
            print(f"Iteration {iteration+1}/{iters}, Validation Accuracy: {accuracy:.4f}")
            
            # After the iteration, compute update norm if not first iteration
            if iteration > 0:
                update_norm = 0.0
                for name, param in model0.named_parameters():
                    update_norm += torch.norm(param.data - prev_params[name].data, p=2).item() ** 2
                update_norm = update_norm ** 0.5  # sqrt to get L2 norm
                update_norms.append(update_norm)
                
            # Update previous parameters for next iteration
            prev_params = {k: v.clone() for k, v in model0.named_parameters()}
        
        # saved the benign model as modelt
        self.modelt_sd = deepcopy(model0.state_dict())
        print(f"Final Validation Accuracy: {vali_accuracies[-1]:.4f}")

        # Step 2: Determine attack direction using negative cross entropy
        self.att_direction, _ = train_one_epoch_with_negative_ce(model0, pooled_data_loader, device=self.device)
        self.benign_direction = {
                                name: self.modelt_sd[name] - self.model0_sd[name] 
                                for name in self.modelt_sd
                                }

        att_norm = get_l2_norm(self.att_direction)
        benign_norm = get_l2_norm(self.benign_direction)

        print(f"L2 norm of att_direction: {att_norm:.3f}")
        print(f"L2 norm of benign_direction: {benign_norm:.3f}")
        
        # Get the last update norm (or 0 if no updates were made)
        self.last_update_norm = update_norms[-1] if update_norms else 0.0

        max_attack_l2_dist, max_attack_acc, attack_accuracy_list, l2_distances = self.find_minmax_l2(test_data)
        modelt_acc = attack_accuracy_list[0]

        return max_attack_l2_dist, max_attack_acc, attack_accuracy_list, l2_distances, modelt_acc


    def find_minmax_l2(self, test_data):
        magnitudes = (
            np.linspace(0, 0.01, 6, endpoint=True).tolist() +
            np.linspace(0.01, 0.1, 6, endpoint=True).tolist() +
            np.linspace(0.1 + 1e-6, 1, 6, endpoint=True).tolist() +
            np.linspace(1, 20, 20, endpoint=True).tolist()
        )
        
        modelt = self.model_fn().to(self.device)
        modelt.load_state_dict(self.modelt_sd)
        c_criterion_t = nn.CrossEntropyLoss()
        attacked_models, l2_distances = self.apply_attack_magnitude(modelt, 
                                                               self.att_direction, 
                                                               self.last_update_norm,
                                                               magnitudes)
    
        attack_accuracy_list = []
        for i, attacked_model in enumerate(attacked_models):
            # test_loss, accuracy, per_class_accuracy, attack_success_rate = fl_validation(
            #     attacked_model, c_criterion_t, pooled_data, server_batch_size=32, class_num=10,
            #     attack_method='none', mali_labels=[0,7], device=self.device
            # )
            test_loss, accuracy, per_class_accuracy = \
                model_validation(model=attacked_model, 
                                 criterion=c_criterion_t, 
                                 test_data=test_data, 
                                 device=self.device, 
                                 class_num=10)
                
            attack_accuracy_list.append(accuracy*100)

        max_att_index = next(
            (i for i, acc in enumerate(attack_accuracy_list) if acc <= 10),
            min(enumerate(attack_accuracy_list), key=lambda x: x[1])[0]
        )
        max_attack_l2_dist = l2_distances[max_att_index]
        
        return max_attack_l2_dist, attack_accuracy_list[max_att_index], attack_accuracy_list, l2_distances
    
    
    def apply_attack_magnitude(self, model, att_direction, average_l2_norm, magnitudes):
        attacked_models = []
        l2_distances = []
        original_params = deepcopy(flatten_model(model.state_dict()))

        for magnitude in magnitudes:
            l2_budget = magnitude * average_l2_norm
            print(f"Applying attack with L2 budget: {l2_budget}")

            malicious_model = apply_attack_to_all_layers(model, att_direction, l2_budget)

            attacked_models.append(malicious_model)
            malicious_params = flatten_model(malicious_model.state_dict())
            distance = torch.norm(original_params - malicious_params, p=2)
            l2_distances.append(distance.item())

        return attacked_models, l2_distances
    
    
    def attack(self):
        """
        Perform the attack on the server model
        """
        device = self.device  # convenience

        # Step 1: Measure progress
        if self.previous_server_model is not None:
            server_update = flatten_model(clone_model_params(self.server_model)).to(device) - \
                            flatten_model(clone_model_params(self.previous_server_model)).to(device)

            server_update_norm = torch.norm(server_update).to(device)

            flat_att_direction = flatten_model(self.att_direction).to(device)
            flat_benign_direction = flatten_model(self.benign_direction).to(device)

            attack_cos_sim = torch.dot(server_update, flat_att_direction) / \
                            (torch.norm(server_update) * torch.norm(flat_att_direction))

            benign_cos_sim = torch.dot(server_update, flat_benign_direction) / \
                            (torch.norm(server_update) * torch.norm(flat_benign_direction))

            if attack_cos_sim / benign_cos_sim <= torch.norm(flat_benign_direction) / torch.norm(flat_att_direction):
                self.scaling_factor *= self.reduce_factor
        else:
            server_update_norm = self.last_update_norm

            flat_att_direction = flatten_model(self.att_direction).to(device)

        print(f"Server update norm: {server_update_norm:.3f}")
        print(f"self.scaling_factor: {self.scaling_factor:.3f}")

        # Step 2: Craft malicious model update
        crafted_flat_update = self.scaling_factor * server_update_norm * \
                            (flat_att_direction / torch.norm(flat_att_direction))

        # Step 3: Convert flat update to state_dict
        crafted_update = {}
        idx = 0
        for name, param in self.server_model.state_dict().items():
            numel = param.numel()
            crafted_update[name] = crafted_flat_update[idx:idx + numel].view_as(param).clone().to(device)
            idx += numel

        # Step 4: Apply crafted update to server model
        attacked_model = deepcopy(self.server_model)
        with torch.no_grad():
            for name, param in attacked_model.named_parameters():
                param.data += crafted_update[name].to(device)

        return crafted_update, attacked_model



def model_validation(model, criterion, test_data, device, class_num):
    """
    Validate the model on the given data loader
    """
    # Evaluate on original test loader for accuracy
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
    # Set up containers
    model.eval()
    correct_pred = {class_id: 0 for class_id in range(class_num)}
    total_pred = {class_id: 0 for class_id in range(class_num)}
    per_class_accuracy = {}
        
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            prediction = prediction.flatten()
            target = target.flatten()
            for class_id, pred in zip(target, prediction):
                total_pred[int(class_id)] += 1
                if class_id == pred:
                    correct_pred[int(class_id)] += 1
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    for class_id in range(class_num):
        if total_pred[class_id] != 0:
            per_class_accuracy[class_id] = correct_pred[class_id] / total_pred[class_id]
        else:
            per_class_accuracy[class_id] = -1

    return test_loss, accuracy, per_class_accuracy


def get_l2_norm(tensor_dict):
    return torch.sqrt(sum((v**2).sum() for v in tensor_dict.values())).item()

"""
# Example usage
config = Config()
train_data, _ = get_dataset(config.dataset)
pooled_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)

# Pretraining to get attack direction
model_fn = partial(get_model(config.model_name), num_classes=config.num_classes, dataset=config.dataset)
initial_weights, attack_direction, last_update_norm = FedOda_pretrain(config, pooled_data_loader, model_fn, config.device)

# Simulate FL process
server_model = model_fn().to(config.device)
previous_server_model = model_fn().to(config.device)
server_model.load_state_dict(initial_weights)
previous_server_model.load_state_dict(initial_weights)

# Craft malicious update
crafted_model = FedOda(server_model, previous_server_model, attack_direction, 
                       flatten_model(attack_direction), 2.0, config.device)
"""