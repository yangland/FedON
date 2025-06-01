import numpy as np
import torchvision.transforms as T
import torch
from datetime import datetime
import logging
from PIL import Image
from scipy.spatial.distance import cdist, euclidean
from wassdistance import SinkhornDistance
from aggr.fedavg_median import fedavg
from collections import OrderedDict
from scipy.stats import norm  # For normal distribution CDF (phi function)
from copy import deepcopy
logger = logging.getLogger("individual_logging")
logger.addHandler(logging.StreamHandler())


def random_recolor(img, background, target_color, random_back=False, random_target=False, output_channel='L'):

    img = img.convert("L")
    d = img.getdata()
    target_image = []
    background_image = []
    img1 = img.convert("RGBA")
    img2 = img.convert("RGBA")

    if random_back:
        background = tuple(np.random.choice(range(256), size=3))
    if random_target:
        target_color = tuple(np.random.choice(range(256), size=3))
    
    for item in d:
        # change black to another color
        target_image.append(background+(item,))
        background_image.append(target_color + (255,))

    img1.putdata(target_image)
    img2.putdata(background_image)
    
    img1.paste(img2, (0,0), mask = img1) 
    
    # img.putdata(new_image)
    img1 = img1.convert(output_channel)

    return img1

def generate_adv_train(adv_train, mult_datasize, channel, train_org):
    if adv_train !='none':
        # process with adverserial training
        logger.info(f"Processing adversarial dataset ({adv_train}) for adversearila training")
        train = {}

        if adv_train in ["invert", "50invert-50normal"]:
            for i in range (len(train_org)):
                X = train_org[i][0]
                y = train_org[i][1]
                
                # 1. invert X
                if adv_train == "invert":
                    X_hat = 1 - X

                # 2. 50% random invert, 50% normal
                if adv_train == "50invert-50normal":
                    if np.random.uniform(0,1) > 0.5:
                        X_hat = 1 - X
                    else:
                        X_hat = X
                
                train[i] = (X_hat, y)
        
        # 3. random recolor, normalize to [0,1]
        if adv_train == 'recolor':
            for j in range(mult_datasize):
                logger.info(f"Multplier data size {j+1}/{mult_datasize}")
                for i in range (len(train_org)):
                    X = train_org[i][0]
                    y = train_org[i][1]
                    transform = T.ToPILImage()
                    X = transform(X)
                    X_hat = random_recolor(X, background=None, target_color=None, random_back=True, 
                                                        random_target=True, output_channel=channel)
                    transform = T.PILToTensor()
                    X_hat = transform(X_hat)/255
                    X_hat = X_hat/torch.max(X_hat)

                    ind = j * len(train_org) + i
                    train[ind] = (X_hat, y)
    else:
        train = train_org
    return train

def invert_color(dataset):
    invert= {}
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        x = 1-x
        invert[i] = (x, y)
    return invert

def timestamp(filename=True):
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y%m%d_%H%M%S")
    if filename:
        return currentTime
    else:
        return currentDateAndTime.strftime("H%:M%:S")
    
def show_img(x_data):
    img = Image.fromarray(np.array(x_data * 255).transpose((1,2,0)).astype(np.uint8))
    img.show()
    return img

def label_mapping(input_tensor, class_label_list, reverse=False):
    class_label_list.sort()
    label_mapping_dict = {class_label_list[i]:i for i in range(len(class_label_list))}
    if reverse:
        label_mapping_dict = {v: k for k, v in label_mapping_dict.items()}
    input_numpy = input_tensor.cpu().numpy()
    output = []
    for i in input_numpy:
        t_i = label_mapping_dict[i]
        output.append(t_i)
    return torch.tensor(output)

# def geometric_median(X, eps=1e-5):
#     X = X.cpu().detach().numpy()
#     y = np.mean(X, 0)

#     while True:
#         D = cdist(X, [y])
#         nonzeros = (D != 0)[:, 0]

#         Dinv = 1 / D[nonzeros]
#         Dinvs = np.sum(Dinv)
#         W = Dinv / Dinvs
#         T = np.sum(W * X[nonzeros], 0)

#         num_zeros = len(X) - np.sum(nonzeros)
#         if num_zeros == 0:
#             y1 = T
#         elif num_zeros == len(X):
#             return y
#         else:
#             R = (T - y) * Dinvs
#             r = np.linalg.norm(R)
#             rinv = 0 if r == 0 else num_zeros/r
#             y1 = max(0, 1-rinv)*T + min(1, rinv)*y

#         if euclidean(y, y1) < eps:
#             return torch.tensor(y1)

#         y = y1


def wass_2_dists(dist1, dist2, device):
    sinkhorn = SinkhornDistance(eps=0.001, max_iter=500, device=device)
    d1_keys = list(dist1.keys())
    d2_keys = list(dist2.keys())
    matrix = []
    for i in d1_keys:
        # print(i,"/",d-1)
        row = []
        for j in d2_keys:
            a_z = dist1[i].to(device)
            b_z = dist2[j].to(device)

            wass, P, C = sinkhorn(a_z, b_z)
            row.append(wass.item())
        matrix.append(row)
    return np.array(matrix)


def training_csv_logging(iter, clients_eval_dict, sr_index, related_candidate_measure, mistake_num, 
                         selection_num, sm_weights, sr_weights, w_sm_sr, clients_weights, avg_resos,
                         candidate_sensors, clients_cross_eval_csv, master_sensor_eval_csv,
                         candidate_sensor_eval_csv, sensor_selection_csv, weights_master_candidate_csv, 
                         representation_weights_csv, mistakes_csv, reso_csv, cross_eval_multi_csv):
    
    # logging into csv files
    for client_id, record in clients_eval_dict.items():
        clients_cross_eval_csv.append_save_csv([iter, client_id] + list(record.values()))
    
    # for class_id, record in multi_dict.items():
    #     cross_eval_multi_csv.append_save_csv([iter, class_id] + list(record.values()))

    sensor_selection_csv.append_save_csv([iter, sr_index] + \
                                            list(related_candidate_measure.values()))

    for client_id, record in sm_weights.items():
        master_sensor_eval_csv.append_save_csv([iter, client_id] + list(record.values()))
    
    for client_id, record in sr_weights.items():
        candidate_sensor_eval_csv.append_save_csv([iter, client_id, candidate_sensors[sr_index].name] + list(record.values()))              
        
    weights_master_candidate_csv.append_save_csv([iter] + w_sm_sr)
    
    for client_id, record in clients_weights.items():
        representation_weights_csv.append_save_csv([iter, client_id] + list(record.values()))   
        
    # mistakes_csv.append_save_csv([iter] + [mistake_num, selection_num])
    
    reso_csv.append_save_csv([iter] + list(avg_resos))
    
    
def lie_attack_old(malicious_client_ids, mco_dict, model_type, Z_max):
    # based on paper: A little is enough: circumventing defenses for distributed learnin
    # https://dl.acm.org/doi/10.5555/3454287.3455062
    # attack method from a little is enough
    # average the malicious clients as mean
    # supporter 20/2 + 1 - 4 = 7
    # Z_max, (20-4-7)/(20-4) = 0.5627 -> Z table 0.16
    # minus Z_max * std from mean

    malicious_models = {
            client_id: mco_dict[client_id][model_type].state_dict() 
            for client_id in malicious_client_ids
        }
    mean = fedavg(malicious_models)  # Must return a state_dict
    attacked_sd = {}
        
    with torch.no_grad():
        for layer_name, param in malicious_models[malicious_client_ids[0]].items():
            if layer_name.endswith('num_batches_tracked'):
                # Preserve as-is
                attacked_sd[layer_name] = malicious_models[malicious_client_ids[0]][layer_name].clone()
                continue

            # Stack values across malicious clients
            layer_values = [malicious_models[client_id][layer_name].float()
                            for client_id in malicious_client_ids]
            stacked = torch.stack(layer_values)
            std = torch.std(stacked, dim=0, correction=0)
            attacked_value = mean[layer_name].float() - Z_max * std

            # Ensure same dtype and shape
            attacked_sd[layer_name] = attacked_value.to(param.dtype)

    return attacked_sd

def lie_attack(malicious_client_ids, mco_dict, model_type, malicious_fraction):
    """
    Perform the LIE attack based on global malicious client percentage.
    
    Args:
        malicious_client_ids: List of malicious client IDs in current round
        mco_dict: Dictionary containing client models in current round
        model_type: Type of model to attack
        malicious_fraction: Global percentage of malicious workers (0-1)
    """
    # Calculate Z_max based on global malicious percentage
    # Calculate needed supporters using formula: (total/2 + 1) - malicious_count
    # Here we use the global malicious percentage to compute expected behavior
    # For the calculation, we assume the same ratio holds in the selected clients

    total = 100
    mali_num = int(total * malicious_fraction)
    supporters_needed = int(total/2 + 1) - mali_num
    phi_z = (total - mali_num - supporters_needed)/(total - mali_num)
    
    # Get Z_max from normal distribution inverse CDF
    try:
        Z_max = abs(norm.ppf(phi_z))
    except:
        # Fallback values
        Z_max = 1.43  # Default value from paper example
    
    logger.info(f"[LIE] based on malicious_fraction: {malicious_fraction}, Z_max: {Z_max:.4f}")
    
    # Get the first malicious model to access its named_parameters
    model = mco_dict[malicious_client_ids[0]][model_type]
    attacked_sd = model.state_dict()  # Start with original state
    
    with torch.no_grad():
        # Get mean and std across all malicious models for each parameter
        param_stats = {}
        for name, param in model.named_parameters():
            # Stack this parameter from all malicious models
            param_values = [mco_dict[client_id][model_type].state_dict()[name].float() 
                          for client_id in malicious_client_ids]
            stacked = torch.stack(param_values)
            param_stats[name] = {
                'mean': torch.mean(stacked, dim=0),
                'std': torch.std(stacked, dim=0, correction=0)
            }

        # Apply attack only to trainable parameters
        for name, param in model.named_parameters():
            stats = param_stats[name]
            attacked_value = stats['mean'] - Z_max * stats['std']
            attacked_sd[name] = attacked_value.to(param.dtype)

    return attacked_sd


class MPAFAttack:
    def __init__(self, model_type="c_model"):
        self.server_sd_init = None
        self.model_type = model_type

    def __call__(self, malicious_client_ids, mco_dict, magnitude=1.0):
        malicious_updates = {}

        # Cache server_sd_init only once (assuming initial weights are identical)
        if self.server_sd_init is None:
            first_client = malicious_client_ids[0]
            self.server_sd_init = {
                key: torch.clone(val).detach()
                for key, val in mco_dict[first_client][self.model_type].state_dict().items()
            }
            logger.info("[MPAF] Cache server_sd_init only once")

        for client_id in malicious_client_ids:
            latest_sd = mco_dict[client_id][self.model_type].state_dict()
            diff = {}

            for key in latest_sd:
                if key.endswith('num_batches_tracked'):
                    diff[key] = torch.clone(self.server_sd_init[key])
                else:
                    diff[key] = magnitude * (latest_sd[key] - self.server_sd_init[key])

            malicious_updates[client_id] = diff

        return malicious_updates



def is_trainable_param(name):
    # Typical buffer keys to exclude
    buffer_keywords = ['running_mean', 'running_var', 'num_batches_tracked']
    return not any(buf in name for buf in buffer_keywords)


def model_weight_diff(new_sd, old_sd, magnitude=1):
    diff = {}
    for key in new_sd.keys():
        if key in old_sd:
            diff[key] = magnitude * (new_sd[key] - old_sd[key])
        else:
            diff[key] = magnitude * new_sd[key]
    return diff

    
def fang_attack(malicious_client_ids, mco_dict, model_sd_before, model_type, device):
    # based on paper: Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
    # particial knowledge attack on Trimmed Mean and Median aggregation rules
    
    malicious_models = {client_id: mco_dict[client_id][model_type].state_dict() for client_id in malicious_client_ids}
    mean_sd = fedavg(malicious_models)
    attacked_sd = {}
    diff = model_weight_diff(mean_sd, model_sd_before)
    
    with torch.no_grad():
        for layer_name, param in malicious_models[malicious_client_ids[0]].items():
            layer_list = []
            for client_id in malicious_client_ids:
                layer_weight = malicious_models[client_id][layer_name]
                layer_list.append(layer_weight)
            stacked_layer_w = torch.stack(layer_list)
            # get standard deviation
            std = torch.std(stacked_layer_w.float(), dim=0, correction=0)
            
            s = torch.sign(diff[layer_name]).to(device)
            layer_size = mean_sd[layer_name].size()
            r1, r2 = 3, 4
            random_mult = (r1 - r2) * torch.rand(layer_size) + r2
            random_mult = random_mult.to(device)
            
            attacked_sd[layer_name] = mean_sd[layer_name].float() - s * random_mult * std.float()       
    
    return attacked_sd


def update_model_with_malicious_gradient(state_dict, server_state, mal_update):
    new_state_dict = {}
    idx = 0

    for name in state_dict:
        if name.endswith('num_batches_tracked'):
            new_state_dict[name] = server_state[name]  # Keep as-is
            continue

        numel = state_dict[name].numel()
        user_grad = mal_update[idx:idx + numel].reshape(state_dict[name].shape)
        new_state_dict[name] = server_state[name] + user_grad
        idx += numel  # Only increment idx for used gradients

    return new_state_dict



def MinMax_attack(malicious_client_ids, mco_dict, model_sd_before, model_type, device, dev_type ='std', threshold=30):
    # from FedREDefense code
    malicious_models = {client_id: mco_dict[client_id][model_type].state_dict() for client_id in malicious_client_ids}
    all_updates = []
    for client_id in malicious_client_ids:
        # get the local weight difference (gradient)
        update_params = get_model_update(malicious_models[client_id], model_sd_before)
        # get the local weight difference (gradient)
        update_params_flat = modelsd2flat(update_params)
        all_updates.append(update_params_flat)
    all_updates = torch.stack(all_updates)
    model_re = torch.mean(all_updates, dim=0).to(device)

    if dev_type == 'unit_vec':
        # unit vector, dir opp to good dir
        deviation = (model_re / torch.norm(model_re)).to(device)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re).to(device) 
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0).to(device)
    lamda = torch.tensor([threshold], dtype=torch.float32, device=device)


    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    if len(all_updates) != 1:
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)

            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        mal_update = (model_re - lamda_succ * deviation)
    else:
        mal_update = (model_re - model_re)
    
    attacked_sd = update_model_with_malicious_gradient(model_sd_before, model_sd_before, mal_update)
    
    return attacked_sd


def MinSum_attack(malicious_client_ids, mco_dict, model_sd_before, model_type, device, dev_type ='std', threshold=30):
    # from FedREDefense code
    malicious_models = {client_id: mco_dict[client_id][model_type].state_dict() for client_id in malicious_client_ids}
    all_updates = []
    for client_id in malicious_client_ids:
        # get the local weight difference (gradient)
        update_params = get_model_update(malicious_models[client_id], model_sd_before)
        # get the local weight difference (gradient)
        update_params_flat = modelsd2flat(update_params)
        all_updates.append(update_params_flat)
    all_updates = torch.stack(all_updates)
    model_re = torch.mean(all_updates, dim=0)

    if dev_type == 'unit_vec':
        # unit vector, dir opp to good dir
        deviation = (model_re / torch.norm(model_re)).to(device)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re).to(device)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    lamda = torch.tensor([threshold], dtype=torch.float32, device=device)


    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    if len(all_updates) != 1:
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        mal_update = (model_re - lamda_succ * deviation)
    # print(lamda_succ)
    else:
        mal_update = (model_re - model_re)
    
    attacked_sd = update_model_with_malicious_gradient(model_sd_before, model_sd_before, mal_update)
    return attacked_sd


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    with torch.no_grad():
        for var in global_parameters:
            if var.split('.')[-1] == 'num_batches_tracked':
                global_parameters[var] = params[0][var]
                continue
            global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


def get_model_update(updated_model, model):
    '''Returns the difference while preserving gradients'''
    update = {}
    for key in updated_model:
        if key.endswith('num_batches_tracked'):
            continue
        update[key] = updated_model[key] - model[key].detach()  # Explicitly detach base model
    return update

def get_model_merged(gradient_update, base_model):
    '''Merges gradient into model while preserving autograd'''
    merged = {}
    for key in base_model:
        if key.endswith('num_batches_tracked'):
            merged[key] = base_model[key]  # Copy directly
        else:
            # Preserve gradient from the update while keeping base model constant
            merged[key] = base_model[key].detach() + gradient_update[key]
    return merged


def modelsd2flat(model_dict):
    """
    Flattens a model's state dictionary into a single 1D tensor.
    Skips non-trainable buffers like 'num_batches_tracked' for consistency.
    """
    ravel_list = []
    for layer_name, parms in model_dict.items():
        if isinstance(parms, torch.Tensor) and not layer_name.endswith('num_batches_tracked'):
            ravel_list.append(parms.detach().reshape(-1))
    
    if not ravel_list:
        return torch.tensor([])
    
    flat_tensor = torch.cat(ravel_list)
    return flat_tensor



def crosseval_high_accuracy_subset(accuracy_matrix): 
    n = len(accuracy_matrix)
    min_subset_size = n // 2 + 1
    accuracy_matrix = np.array(accuracy_matrix)
    subset = set(range(n))
    
    # Precompute total pairwise accuracies and counts for all pairs
    total_pairwise_accuracy = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            total_pairwise_accuracy[i, j] = accuracy_matrix[i, j] + accuracy_matrix[j, i]
            total_pairwise_accuracy[j, i] = total_pairwise_accuracy[i, j]
    
    # Initialize subset statistics
    current_total = np.sum(total_pairwise_accuracy) / 2  # Sum of upper triangle
    current_count = n * (n - 1)  # Total number of pairwise comparisons (both directions)
    
    def calculate_avg(total, count):
        return total / count if count > 0 else 0

    while len(subset) > min_subset_size:
        best_avg = -1
        best_to_remove = None
        
        for candidate in list(subset):
            temp_total = current_total - np.sum(total_pairwise_accuracy[candidate, list(subset - {candidate})])
            temp_count = current_count - 2 * (len(subset) - 1)
            current_avg = calculate_avg(temp_total, temp_count)
            
            if current_avg > best_avg or (current_avg == best_avg and candidate < best_to_remove):
                best_avg = current_avg
                best_to_remove = candidate
        
        # Update statistics by removing the selected client
        current_total -= np.sum(total_pairwise_accuracy[best_to_remove, list(subset - {best_to_remove})])
        current_count -= 2 * (len(subset) - 1)
        
        subset.remove(best_to_remove)
    
    return sorted(subset)


def selfeval_subset(accuracy_matrix, multi=3):
    std_dev = np.std(accuracy_matrix)
    mean = np.mean(accuracy_matrix)
    subset = list(range(len(accuracy_matrix)))
    self_eval = np.diag(accuracy_matrix)
    for i in range(len(self_eval)):
        if self_eval[i] < mean - multi * std_dev:
            subset.remove(i)
    return subset, mean, std_dev


def create_client_weights(client_selection, benign_client_ids, client_list):
    """
    Creates client weight dictionaries indicating benign (1) and non-benign (0) clients.
    
    Args:
        client_selection: List of all client IDs
        benign_idx: Indices of benign clients in client_selection
        client_list: List of all client IDs to consider
        
    Returns:
        clients_weights_c: Dictionary mapping client IDs to weights (0 or 1)
        clients_weights_g: Same as clients_weights_c (copied)
    """
    
    clients_weights_c = {}
    
    for client_id in client_list:
        if client_id not in benign_client_ids:
            clients_weights_c[client_id] = 0        
        else:
            clients_weights_c[client_id] = 1
    
    clients_weights_g = clients_weights_c.copy()
    
    return clients_weights_c, clients_weights_g



def flat2modelsd(flat_tensor, model_sd_template):
    new_sd = {}
    idx = 0
    for k, v in model_sd_template.items():
        if v.dtype == torch.long or 'num_batches_tracked' in k:
            new_sd[k] = v
            continue
        numel = v.numel()
        new_sd[k] = flat_tensor[idx:idx+numel].view_as(v)
        idx += numel
    return new_sd

def grad_normlization(client_grad_dict):
    flat_grads = {}
    norms = []

    # 1. Flatten and compute norm
    for client_id, grad_dict in client_grad_dict.items():
        flat = modelsd2flat(grad_dict)
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.norm(flat, p=2)
        flat_grads[client_id] = flat
        norms.append(norm.item())

    # 2. Compute median of norms
    norms = np.array(norms)
    norms = norms[~np.isnan(norms)]
    if len(norms) == 0:
        raise ValueError("All norms are NaN. Check the input gradients.")
    clip_value = np.median(norms)

    # 3. Normalize (clip if norm > median) and reconstruct state_dict
    c_grad_dict_norm = {}
    for client_id, flat in flat_grads.items():
        norm = torch.norm(flat, p=2).item()
        gamma = clip_value / (norm + 1e-15)
        if gamma < 1:
            flat = flat * gamma
        clipped_sd = flat2modelsd(flat, client_grad_dict[client_id])
        c_grad_dict_norm[client_id] = clipped_sd

    return clip_value, c_grad_dict_norm


def grad_normlization_factor(client_grad_dict):
    flat_grads = {}
    norms = []

    # 1. Flatten and compute norm
    for client_id, grad_dict in client_grad_dict.items():
        flat = modelsd2flat(grad_dict)
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.norm(flat, p=2)
        flat_grads[client_id] = flat
        norms.append(norm.item())

    # 2. Compute median of norms
    norms = np.array(norms)
    norms = norms[~np.isnan(norms)]
    if len(norms) == 0:
        raise ValueError("All norms are NaN. Check the input gradients.")
    clip_value = np.median(norms)

    # 3. Compute scaling weights for each client
    weights = {}
    for client_id, flat in flat_grads.items():
        norm = torch.norm(flat, p=2).item()
        weight = clip_value / (norm + 1e-15)
        weights[client_id] = weight

    return clip_value, weights


def check_model_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[NaN DETECTED] in parameter: {name}")

def check_bn_buffers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if torch.isnan(module.running_mean).any() or torch.isnan(module.running_var).any():
                print(f"[NaN DETECTED] in BatchNorm buffer: {name}")

def safe_tensor(tensor):
    tensor = tensor.clone()
    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0
    return tensor

def safe_model_state_dict(state_dict):
    safe_sd = {}
    for name, param in state_dict.items():
        if torch.is_tensor(param):
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[Warning] Detected NaN/Inf in layer: {name}")
            safe_sd[name] = safe_tensor(param)
        else:
            safe_sd[name] = param
    return safe_sd


def average_tensor_dict(tensor_dicts):
    """Averages dicts of gradients (same format as state_dicts)"""
    tensor_list = list(tensor_dicts.values())
    avg = deepcopy(tensor_list[0])
    for k in avg:
        avg[k] = torch.stack([t[k] for t in tensor_list]).mean(0)
    return avg