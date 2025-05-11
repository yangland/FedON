import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from run_intro_exp import (
    get_dataset,
    train_one_epoch_with_negative_ce,
    apply_attack_to_all_layers,
    calculate_distance_matrix,
    flatten_model,
    clone_model_params
)

def FedOda_pretrain(config, pooled_data_loader, model_fn, device):
    # Step 1: Initialize models
    model0 = model_fn().to(device)
    modelt = model_fn().to(device)
    modelt.load_state_dict(model0.state_dict())

    # Train model0 to convergence (simulate this part as per your setup)
    optimizer0 = optim.SGD(model0.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Here, you would train model0 on the pooled_data_loader until convergence
    # Simulate with a single epoch for demonstration purposes
    for _ in range(config.iters):
        for inputs, labels in pooled_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer0.zero_grad()
            outputs = model0(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer0.step()

    # Step 2: Determine attack direction using negative cross entropy
    G_m, _ = train_one_epoch_with_negative_ce(modelt, pooled_data_loader, device)

    # Compute L2 norm of the final update
    last_update_norm = torch.norm(flatten_model(G_m))

    return model0.state_dict(), G_m, last_update_norm


def FedOda(server_model, previous_server_model, attack_direction, benign_direction, scaling_factor, reduce_factor, device):
    # Step 1: Measure progress
    server_update = flatten_model(clone_model_params(server_model)) - flatten_model(clone_model_params(previous_server_model))
    attack_cos_sim = torch.dot(server_update, attack_direction) / (torch.norm(server_update) * torch.norm(attack_direction))
    benign_cos_sim = torch.dot(server_update, benign_direction) / (torch.norm(server_update) * torch.norm(benign_direction))

    if attack_cos_sim / benign_cos_sim <= torch.norm(benign_direction) / torch.norm(attack_direction):
        scaling_factor *= reduce_factor  # Reduce scaling factor if attack progress is insufficient

    # Step 2: Craft malicious model update
    server_update_norm = torch.norm(server_update)
    crafted_update = scaling_factor * server_update_norm * (attack_direction / torch.norm(attack_direction))

    # Apply crafted update to server model
    attacked_model = deepcopy(server_model)
    with torch.no_grad():
        for name, param in attacked_model.named_parameters():
            param.data += crafted_update[name].to(param.device)

    return attacked_model

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