import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
from helper import fl_validation
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
    def __init__(self, model_fn, device, learning_rate=0.01, reduce_factor=0.5):
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
        self.scaling_factor = 2 # Current scaling factor
        self.server_model = None  # Server model
        self.previous_server_model = None
        
    def pretrain(self, iters, pooled_data):
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
        
        for iteration in range(iters):
            for inputs, labels in pooled_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer_t.zero_grad()
                outputs = model0(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_t.step()

            # After the iteration, compute update norm if not first iteration
            if iteration > 0:
                update_norm = 0.0
                for name, param in model0.named_parameters():
                    update_norm += torch.norm(param.data - prev_params[name].data, p=2).item() ** 2
                update_norm = update_norm ** 0.5  # sqrt to get L2 norm
                update_norms.append(update_norm)
                
            # Update previous parameters for next iteration
            prev_params = {k: v.clone() for k, v in model0.named_parameters()}

        self.modelt_sd = deepcopy(model0.state_dict())

        # Step 2: Determine attack direction using negative cross entropy
        self.att_direction, _ = train_one_epoch_with_negative_ce(model0, pooled_data_loader)
        self.benign_direction = (self.modelt_sd - self.model0_sd)

        # Get the last update norm (or 0 if no updates were made)
        self.last_update_norm = update_norms[-1] if update_norms else 0.0

        max_attack_l2_dist, max_attack_acc, attack_accuracy_list, l2_distances = self.find_minmax_l2(pooled_data)
        modelt_acc = attack_accuracy_list[-1]

        return max_attack_l2_dist, max_attack_acc


    def find_minmax_l2(self, pooled_data):
        magnitudes = (
            np.linspace(0, 0.01, 6, endpoint=True).tolist() +
            np.linspace(0.01, 0.1, 6, endpoint=True).tolist() +
            np.linspace(0.1 + 1e-6, 1, 6, endpoint=True).tolist() +
            np.linspace(1, 20, 20, endpoint=True).tolist()
        )
        
        modelt = self.model_fn().to(self.device)
        modelt.load_state_dict(self.model0_sd)
        c_criterion_t = nn.CrossEntropyLoss()
        attacked_models, l2_distances = self.apply_attack_magnitude(modelt, 
                                                               self.att_direction, 
                                                               self.last_update_norm,
                                                               magnitudes)
    
        attack_accuracy_list = []
        for i, attacked_model in enumerate(attacked_models):
            test_loss, accuracy, per_class_accuracy, attack_success_rate = fl_validation(
                attacked_model, c_criterion_t, pooled_data, server_batch_size=32, class_num=10,
                attack_method='none', mali_labels=[0,7], device=self.device
            )
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
        # Store the current scaling factor and benign direction
        self.benign_direction = self.benign_direction

        # Step 1: Measure progress
        server_update = flatten_model(clone_model_params(self.server_model)) - \
                      flatten_model(clone_model_params(self.previous_server_model))
        
        attack_cos_sim = torch.dot(server_update, self.att_direction) / \
                        (torch.norm(server_update) * torch.norm(self.att_direction))
        benign_cos_sim = torch.dot(server_update, self.benign_direction) / \
                        (torch.norm(server_update) * torch.norm(self.benign_direction))

        if attack_cos_sim / benign_cos_sim <= torch.norm(self.benign_direction) / torch.norm(self.att_direction):
            self.scaling_factor *= self.reduce_factor  # Reduce scaling factor

        # Step 2: Craft malicious model update
        server_update_norm = torch.norm(server_update)
        crafted_update = self.scaling_factor * server_update_norm * \
                       (self.att_direction / torch.norm(self.att_direction))

        # Apply crafted update to server model
        attacked_model = deepcopy(self.server_model)
        with torch.no_grad():
            for name, param in attacked_model.named_parameters():
                param.data += crafted_update[name].to(param.device)

        return crafted_update, attacked_model


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