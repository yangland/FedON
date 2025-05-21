
import copy
import logging
logger = logging.getLogger("individual_logging")
from torch.utils.data import ConcatDataset, DataLoader, Subset
from data_process import data_process, quantity_based_label_dist_skew
import torch
import random
import numpy as np
from collections import defaultdict, Counter
import math
from torch import nn
from functools import partial
from models.model_utils import get_model
from utilities import (
    MinMax_attack,
    MinSum_attack,
    fang_attack,
    label_mapping,
    lie_attack,
    model_weight_diff,
    training_csv_logging,
    create_client_weights,
    grad_normlization,
    grad_normlization_factor
)

from models.ch_mnist import Chmnist_13L
from models.cifar10_18l import Cifar10_18L
from models.convnet import ConvNet, channel_dict
from models.mnist_6l import Mnist_6L
from models.model_utils import get_model
import torch.nn as nn
import torch.nn.functional as F


def fl_iteration(iter, client_selection, malicious_client_ids, attack_method, c_epochs, 
                    fl_data_dict, mco_dict, batch_size, 
                    dba_poisoning_per_batch=0.2, FedOda_class=None, device=None):

    if attack_method == 'FedOda' and bool(set(client_selection) & set(malicious_client_ids)):
        if FedOda_class.server_model is None:
            FedOda_class.server_model = copy.deepcopy(mco_dict[-1]["c_model"])
        else:
            FedOda_class.previous_server_model = copy.deepcopy(FedOda_class.server_model)
            FedOda_class.server_model = copy.deepcopy(mco_dict[-1]["c_model"])

        FedOda_class.current_s_m = len(set(client_selection) & set(malicious_client_ids))
    
    c_sd_before = train_c_models(client_selection, 
                                    fl_data_dict,
                                    batch_size, 
                                    mco_dict, 
                                    c_epochs,
                                    attack_method,
                                    malicious_client_ids,
                                    dba_poisoning_per_batch,
                                    device=device)
    
    c_grad_dict = {}

    # process for both benign and malicious clients
    c_grad_dict = grad_process_c_model(mco_dict, client_selection, c_sd_before,
                                            malicious_client_ids, attack_method,
                                            FedOda_class, device)

    return c_grad_dict



def train_c_models(client_selection, fl_data_dict, batch_size, mco_dict, c_epochs,
                     attack_method=None, malicious_client_ids=[], dba_poisoning_per_batch=0.2, device=None):
    c_sd_before = {}

    for client_id in client_selection:
        c_sd_before[client_id] = copy.deepcopy(mco_dict[client_id]["c_model"].state_dict())
        client_data = fl_data_dict[client_id]
        client_dl = DataLoader(client_data, batch_size, shuffle=True, num_workers=5, drop_last=True)

        # train the classifier model
        # for dba attack
        optimizer_poison = torch.optim.SGD(mco_dict[client_id]["c_model"].parameters(), lr=0.005)
        
        for epoch in range(c_epochs):
            loss, acc = train_c_model(client_dl, mco_dict[client_id]["c_model"], 
                                        mco_dict[client_id]["c_criterion"], 
                                        mco_dict[client_id]["c_optimizer"], 
                                        attack_method, client_id, 
                                        malicious_client_ids,
                                        optimizer_poison,
                                        dba_poisoning_per_batch,
                                        device=device)
            if epoch % 1 == 0:
                logger.info(f"  client{client_id}_ep{epoch}: C-Model, loss: {loss:.6f}, train acc: {acc:.6f}")
        

    return c_sd_before


attack_magnitude_mapping = {
    '2_label_flipping': 3,
    'MPAF': 3, # 1e6, 0
    'backdoor': 3,
    'label_perturbation': 3,
    'fakedata': 1,
    'clear_label_backdoor': 1,
    'reverse_grad': 1,
    'fang': 1,
    'MinMax': 1,
    'MinSum': 1,
    'LIE': 1,
    'dba': 3,
    'FedOda': 1
}

def grad_process_c_model(mco_dict, client_selection, c_sd_before, malicious_client_ids, attack_method,
                         FedOda_class, device):
    c_grad_dict = {}

    # benign clients
    benign_client_ids = list(set(client_selection).difference(malicious_client_ids))
    for client_id in benign_client_ids:
        c_grad_dict[client_id] = model_weight_diff(mco_dict[client_id]["c_model"].state_dict(), c_sd_before[client_id])
    
    malicious_client_ids = list(set(malicious_client_ids).intersection(client_selection))
    
    # attack_magnitude affecting the c model
    attack_magnitude = attack_magnitude_mapping.get(attack_method, 1)
    
    if attack_method == 'FedOda':
        if FedOda_class is not None:
            FedOda_attack = FedOda_class.attack()
        else:
            raise ValueError("FedOda_class is not initialized.")
    
    # malicous clients
    for client_id in malicious_client_ids:
        if attack_method == 'reverse_grad':
            # reverse the order in the model_weight_diff
            c_grad_dict[client_id] = model_weight_diff(c_sd_before[client_id], mco_dict[client_id]["c_model"].state_dict(), attack_magnitude)
        elif attack_method in ['backdoor', 'fakedata', 'label_perturbation', '2_label_flipping', 'clear_label_backdoor', 'dba']:
            c_grad_dict[client_id] = model_weight_diff(mco_dict[client_id]["c_model"].state_dict(), c_sd_before[client_id], attack_magnitude)
        elif attack_method == 'LIE':
            c_grad_dict[client_id] = lie_attack(malicious_client_ids, mco_dict, model_type="c_model", Z_max=0.16)
        elif attack_method == 'fang':
            c_grad_dict[client_id] = fang_attack(malicious_client_ids, mco_dict, c_sd_before[client_id], model_type="c_model", device=device)
        elif attack_method == 'MinMax':
            c_grad_dict[client_id] = MinMax_attack(malicious_client_ids, mco_dict, c_sd_before[client_id], model_type="c_model", device=device)
        elif attack_method == 'MinSum':
            c_grad_dict[client_id] = MinSum_attack(malicious_client_ids, mco_dict, c_sd_before[client_id], model_type="c_model", device=device)
        elif attack_method == 'MPAF':
            c_grad_dict[client_id] = model_weight_diff(c_sd_before[client_id], 
                                                       mco_dict[client_id]["c_model"].state_dict(), 
                                                       attack_magnitude)
        elif attack_method == 'FedOda':
            c_grad_dict[client_id], _ = FedOda_attack

        
        mco_dict[client_id]["c_model"].load_state_dict(c_sd_before[client_id])
        grad_add_to_model(mco_dict[client_id]["c_model"], lr=1, grad=c_grad_dict[client_id])
    return c_grad_dict


def create_data_fl_clients(client_num, noniid_type, label_skew_alpha, p_degree, dirichlet_alpha, fl_train, fl_class_num):
    logger.info(f"non-iid type: {noniid_type}")
    label_count_per_client_dict = None
    if noniid_type == 'label_skew':
        fl_data_dict, labels_per_client_dict = label_skew_non_iid(fl_class_num, fl_train, client_num, label_skew_alpha)
    elif noniid_type == 'p-degree':
        client_group = p_degree_grouping(fl_class_num, client_num)
        fl_data_dict = p_degree_non_iid(client_group, p_degree, fl_class_num, client_num, fl_train)
        labels_per_client_dict = {x:list(range(fl_class_num)) for x in list(range(client_num))}
    elif noniid_type == 'dirichlet':
        fl_data_dict, labels_per_client_dict, label_count_per_client_dict = dirichlet_non_iid(fl_class_num, client_num, fl_train, dirichlet_alpha) 
    return fl_data_dict, labels_per_client_dict, label_count_per_client_dict


def p_degree_grouping(fl_class_num, client_num):
    # randomly group clients
    clients = list(range(client_num))
    assert client_num % fl_class_num == 0, "for p-degree_non_iid, client num has to be mulipler of fl class num"
    random.shuffle(clients)
    if client_num % fl_class_num == 0:
        group_size = int(client_num/fl_class_num)
    else:
        return -1
    client_group = {}
    for group_id in range(fl_class_num):
        client_group[group_id] = clients[:group_size]
        clients = clients[group_size:]
    return client_group


def p_degree_non_iid(client_group, p_degree, fl_class_num, client_num, fl_train):
    fl_data_dict = {}
    label_indices = {}
    label_counts = {}
    train_targets = [x[1] for x in fl_train]
    clients_per_group = int(client_num/fl_class_num)
    
    for group_class_id in range(fl_class_num):
        label_indices[group_class_id] = [idx for idx, target in enumerate(train_targets) if target == group_class_id]
        label_counts[group_class_id] = len(label_indices[group_class_id])
        logger.info(f"{group_class_id} label_indices: {label_counts[group_class_id]}")
    
    for group_class_id, client_ids in client_group.items():
        for client_id in client_ids:
            fl_data_dict[client_id] = []
            client_label_count = {}
            for fl_class_id in range(fl_class_num):
                if fl_class_id == group_class_id:
                    sample_size = math.floor(p_degree * label_counts[group_class_id]/clients_per_group)
                else:
                    sample_size = math.floor((1-p_degree)/(fl_class_num-1) * label_counts[group_class_id]/clients_per_group)
                client_label_count[fl_class_id] = sample_size
                x_indices = label_indices[fl_class_id][:sample_size]
                label_indices[fl_class_id] = label_indices[fl_class_id][sample_size:]
                
                for j in x_indices:
                    fl_data_dict[client_id].append([fl_train[j][0], fl_class_id])
                
            logger.info(f"fl_data client{client_id} size:{len(fl_data_dict[client_id])}: {client_label_count}")
    return fl_data_dict


def label_skew_non_iid(fl_class_num, fl_train, client_num, label_skew_alpha):
    fl_data_dict = {}
    train_targets = [x[1] for x in fl_train]
    labels_per_client_dict, label_counts = quantity_based_label_dist_skew(fl_class_num, client_num, label_skew_alpha)
    label_num_avg, label_num_std = np.mean(list(label_counts.values())), np.std(list(label_counts.values()))
    
    sample_idx = {x: 0 for x in range(fl_class_num)}
    
    for i in range(client_num):
        fl_data_dict[i] = []
        for label in labels_per_client_dict[i]:
            label_indices = [idx for idx, target in enumerate(train_targets) if target == label]
            # use the most common label as data amount
            sample_size = int(len(label_indices) / max(label_counts.values()))
            x_indices = label_indices[sample_size*sample_idx[label] : sample_size*(sample_idx[label]+1)]
            sample_idx[label]+=1

            for j in x_indices:
                fl_data_dict[i].append([fl_train[j][0], label])
        logger.info(f"fl_data_dict{i}, data size {len(fl_data_dict[i])}")
        
    logger.info(f"created label skew dataset, labels for class {labels_per_client_dict}")
    logger.info(f"label_counts for all clients: {label_counts}")
    logger.info(f"per class label clients num: mean {label_num_avg}; std {label_num_std}")
    
    return fl_data_dict, labels_per_client_dict



def dirichlet_non_iid(fl_class_num, client_num, fl_train, dirichlet_alpha):
    from collections import defaultdict
    import numpy as np
    import random

    # Group indices by label
    label_indices = [[] for _ in range(fl_class_num)]
    for idx, (_, label) in enumerate(fl_train):
        label_indices[label].append(idx)

    # Dicts to store client data
    fl_data_dict = defaultdict(list)
    labels_per_client_dict = {x: list(range(fl_class_num)) for x in range(client_num)}
    label_count_per_client_dict = {x: defaultdict(int) for x in range(client_num)}

    total_samples_per_client = np.zeros(client_num, dtype=int)

    for class_id in range(fl_class_num):
        class_indices = label_indices[class_id]
        np.random.shuffle(class_indices)

        # Sample once from Dirichlet
        proportions = np.random.dirichlet([dirichlet_alpha] * client_num)
        proportions = proportions / proportions.sum()  # normalize again in case of precision error
        split_indices = (proportions * len(class_indices)).astype(int)

        # Correct rounding errors to match total count
        diff = len(class_indices) - split_indices.sum()
        for i in range(diff):
            split_indices[i % client_num] += 1

        start = 0
        for client_id, count in enumerate(split_indices):
            end = start + count
            for idx in class_indices[start:end]:
                fl_data_dict[client_id].append([fl_train[idx][0], class_id])
                label_count_per_client_dict[client_id][class_id] += 1
                total_samples_per_client[client_id] += 1
            start = end

    # Check for empty clients and redistribute if necessary
    empty_clients = np.where(total_samples_per_client == 0)[0]
    if len(empty_clients) > 0:
        logger.warning(f"{len(empty_clients)} clients received no data, redistributing randomly.")
        # Flatten all data
        all_data = []
        for client_data in fl_data_dict.values():
            all_data.extend(client_data)
        random.shuffle(all_data)

        # Reinitialize dictionaries
        fl_data_dict = defaultdict(list)
        label_count_per_client_dict = {x: defaultdict(int) for x in range(client_num)}
        per_client = len(all_data) // client_num

        for i, sample in enumerate(all_data):
            cid = i % client_num
            fl_data_dict[cid].append(sample)
            label_count_per_client_dict[cid][sample[1]] += 1

    # Convert label counts to regular dicts
    label_count_per_client_dict = {cid: dict(cnts) for cid, cnts in label_count_per_client_dict.items()}

    logger.info(f"Created Dirichlet non-IID dataset with alpha={dirichlet_alpha}")
    logger.info("Label counts per client:")
    for client_id, counts in list(label_count_per_client_dict.items())[:5]:
        logger.info(f"Client {client_id}: {counts}")
    for client_id, counts in list(label_count_per_client_dict.items())[-5:]:
        logger.info(f"Client {client_id}: {counts}")

    return fl_data_dict, labels_per_client_dict, label_count_per_client_dict




def count_labels(pooled_ds):
    """
    Count labels in a list of datasets or a single dataset.

    Args:
        pooled_ds (Dataset or list of Datasets): The data to count labels from.

    Returns:
        dict: label -> count
    """
    label_counter = Counter()

    # If it's a list of datasets, iterate through each
    if isinstance(pooled_ds, list):
        for dataset in pooled_ds:
            for _, label in dataset:
                # Convert tensor labels to int
                if isinstance(label, torch.Tensor):
                    label = label.item()
                label_counter[label] += 1
    else:
        # Single dataset (e.g., ConcatDataset)
        for _, label in pooled_ds:
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_counter[label] += 1

    return dict(label_counter)


def create_malicious_data(client_num, malicious_num, fl_data_dict, attack_method, image_size, class_num):
    # first m clients are malicious
    malicious_client_ids = list(range(malicious_num))
    logger.info(f"malicious clients: {malicious_num} mali client ({list(malicious_client_ids)}) of total {client_num} clients")
    
    # selecte common label in malicous clients to attack
    # common_labels = find_common_labels(malicious_client_ids, fl_data_dict)
    # # print("common_labels", common_labels)
    # if len(common_labels)>=2:
    #     m_label1, m_label2 = random.sample(common_labels, 2)
    # elif len(common_labels)==1:
    #     m_label1 = list(common_labels)[0]
    #     m_label2 = random.sample([ i for i in list(range(class_num)) if i != m_label1], 1)[0]
    # else:
    #     m_label1, m_label2 = random.sample(list(range(class_num)), 2)

    m_label1, m_label2 = 0, 7
    
    # randommly select label to attack
    # m_label1, m_label2 = random.sample(list(range(class_num)), 2)
    fl_data_dict2 = copy.deepcopy(fl_data_dict)
    
    mali_labels = []

    for client_id in malicious_client_ids:
        ds = fl_data_dict2[client_id]
        ind_label1 = [idx for idx, target in enumerate(ds) if ds[idx][1] == m_label1]
        ind_label2 = [idx for idx, target in enumerate(ds) if ds[idx][1] == m_label2]
        
        if attack_method == 'fakedata':
            sample_size = len(ds)
            if_gray = True if image_size[0]==1 else False
            fakedata, _ = data_process('fake', image_size, grayscal=if_gray) #TODO fix grayscal

            indices = random.sample(list(range(len(fakedata))), sample_size)
            x_fakedata = [x[0] for x in Subset(fakedata, indices)]
            for j in range(len(ds)):
                ds[j][0] = x_fakedata[j] # [1, 28, 28]
            mali_labels = list(range(class_num))
        elif attack_method == '2_label_flipping':
            for k in ind_label1:
                ds[k][1] = m_label2
            for k in ind_label2:
                ds[k][1] = m_label1
            mali_labels = [m_label1, m_label2]
        elif attack_method == 'label_perturbation':
            # follows the label flipping methods in FLTrust 
            # all class label mis match: 0->9, 1->8 ... 9->0
            for i in range(len(ds)):
                ds[i][1] = class_num-1-ds[i][1]
            mali_labels = list(range(class_num))
        elif attack_method == 'backdoor':
            for k in ind_label1:
                ds[k][1] = m_label2
                ds[k][0] = add_trigger( ds[k][0], 
                                        pattern_size=3, 
                                        rgb=[1, 1, 1],
                                        solid=False)
                
            mali_labels = [m_label1, m_label2]
        elif attack_method == 'clear_label_backdoor':
            for k in ind_label1:
                ds[k][0] = add_trigger( ds[k][0], 
                                        pattern_size=3, 
                                        rgb=[1, 1, 1],
                                        solid=False)
                
            mali_labels = [m_label1, m_label2]
        elif attack_method == 'dba':
            """            
            # code from  FedREDefense - utils train_op_dba
            for i in range(len(ds)):
                # half dataset is backdoored
                if i%2 == 0:
                    # change all label to m_label2
                    ds[i][1] = m_label2
                    if client_id %4 == 0:
                        ds[i][0][:,26,26] = trigger_value
                    elif client_id %4 == 1:
                        ds[i][0][:,24,26] = trigger_value
                    elif client_id %4 == 2:
                        ds[i][0][:,26,24] = trigger_value
                    elif client_id %4 == 3:
                        ds[i][0][:,25,25] = trigger_value
                else:
                    # The other half dataset is normal
                    pass
            """
            
            # trigger_value = 1
            # for k in ind_label1:
            #     ds[k][1] = m_label2
            #     if client_id %4 == 0:
            #         ds[k][0][:,26,26] = trigger_value
            #     elif client_id %4 == 1:
            #         ds[k][0][:,24,26] = trigger_value
            #     elif client_id %4 == 2:
            #         ds[k][0][:,26,24] = trigger_value
            #     elif client_id %4 == 3:
            #         ds[k][0][:,25,25] = trigger_value
            
            # DBA need to add trigger during training with batch
            mali_labels = [m_label1, m_label2]  
        elif attack_method == 'none':
            mali_labels = []
        elif attack_method in ['reverse_grad', 'fang', 'LIE', 'MPAF', 'MinMax', 'MinSum', 'FedOda']:
            mali_labels = list(range(class_num))
        else:
            print("attack_method unknown")
            mali_labels = []
        
    return fl_data_dict2, malicious_client_ids, mali_labels


def find_common_labels(client_ids, fl_data_dict):
    client_labels = []
    for client_id in client_ids:
        labels = set([x[1] for x in fl_data_dict[client_id]])
        client_labels.append(labels)
    common_labels = set(client_labels[0]).intersection(*client_labels)
    return common_labels


def add_trigger(img_tensor, pattern_size, rgb, solid):
    # print("img_tensor", img_tensor)
    # print("image.size()", image.size()) # [1,28,28]
    img_size = img_tensor.size()

    for i in range(img_size[0]):
        if solid:
            img_tensor[i, -pattern_size:, -pattern_size:] = rgb[i]        
        else:    
            count = 0
            area = img_tensor[i, -pattern_size:, -pattern_size:]
            for x in range(pattern_size):
                for y in range(pattern_size):
                    if count%2 == 0:
                        area[x,y] = rgb[i]
                    count+=1
    # print("added backdoor", img_tensor)
    return img_tensor


def create_fl_models(client_num, device, c_model_name, c_lr, server_optim_c_lr, c_momentum, num_classes, dataset):
    mco_dict = {}
    for i in range(-1, client_num):
        # C MODEL
        c_model_class = get_model(c_model_name)
        c_model_fn = partial(c_model_class, num_classes=num_classes, dataset=dataset)
        c_model = c_model_fn().to(device)
        
        c_criterion = nn.CrossEntropyLoss()
        
        if i == -1:
            c_optimizer = torch.optim.SGD(c_model.parameters(),
                        lr=server_optim_c_lr, momentum=c_momentum)
        else:
            c_optimizer = torch.optim.SGD(c_model.parameters(),
                            lr=c_lr, momentum=c_momentum)


        mco_dict[i] = {"c_model":c_model, 
                       "c_criterion": c_criterion, 
                       "c_optimizer": c_optimizer}
    
    return mco_dict


def send_main_to_clients(client_num, mco_dict, c_g=[True, True]):
    with torch.no_grad():
        for i in range(client_num):
                mco_dict[i]["c_model"].load_state_dict(mco_dict[-1]["c_model"].state_dict())
    return mco_dict

def train_c_model(train_loader, model, criterion, optimizer, attack_method, client_id,
                  malicious_client_ids, optimizer_poison=None, dba_poisoning_per_batch=0.2, device=None):
    # train the model for 1 epoch
    model = model.to(device)
    model.train()
    train_loss = 0.0
    correct = 0
    trigger_value = 1
    
    if attack_method == 'dba' and client_id in malicious_client_ids:
        print('dba client id', client_id)

        # Poisoned training
        for data, target in train_loader:
            data = data.clone()
            target = target.clone()

            for i in range(len(data)):
                if i < dba_poisoning_per_batch * len(data):
                    coords = [(26,26), (24,26), (26,24), (25,25)][client_id % 4]
                    data[i][:, coords[0], coords[1]] = trigger_value
                    target[i] = 0

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            if optimizer_poison is None:
                raise ValueError("optimizer_poison must be provided for poisoned training")

            optimizer_poison.zero_grad()
            loss.backward()
            optimizer_poison.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        # Clean training
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    else:
        # normal training
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)       

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()    
    
    return train_loss / len(train_loader), correct / len(train_loader.dataset)



def fl_validation(model, criterion, fl_test, server_batch_size, class_num, attack_method, mali_labels, device=None):
    original_test_loader = DataLoader(fl_test, server_batch_size, shuffle=True, num_workers=5)
    test_loader = original_test_loader  # default to original if no attack applies
    
    # Move model to device
    model = model.to(device)
    
    if attack_method == 'backdoor':
        test_targets = [x[1] for x in fl_test]
        indices = [idx for idx, target in enumerate(test_targets) if target == mali_labels[0]]
        backdoored_data = [x for x in Subset(fl_test, indices)]
        for ind in range(len(backdoored_data)):
            x = add_trigger(backdoored_data[ind][0], pattern_size=3, rgb=[1, 1, 1], solid=False)
            backdoored_data[ind] = (x, backdoored_data[ind][1])
        other_indices = list(set(range(len(test_targets))) - set(indices))
        other_data = [x for x in Subset(fl_test, other_indices)]
        modified_test_set = ConcatDataset([backdoored_data, other_data])
        test_loader = DataLoader(modified_test_set, server_batch_size, shuffle=True, num_workers=5)

    elif attack_method == 'dba':
        test_targets = [x[1] for x in fl_test]
        indices = [idx for idx, target in enumerate(test_targets) if target != mali_labels[0]]
        backdoored_data = [x for x in Subset(fl_test, indices)]
        for ind in range(len(backdoored_data)):
            backdoored_data[ind][0][:,26,26] = 1
            backdoored_data[ind][0][:,24,26] = 1
            backdoored_data[ind][0][:,26,24] = 1
            backdoored_data[ind][0][:,25,25] = 1
        other_indices = list(set(range(len(test_targets))) - set(indices))
        other_data = [x for x in Subset(fl_test, other_indices)]
        modified_test_set = ConcatDataset([backdoored_data, other_data])
        test_loader = DataLoader(modified_test_set, server_batch_size, shuffle=True, num_workers=5)

    elif attack_method == 'clear_label_backdoor':
        test_targets = [x[1] for x in fl_test]
        indices = [idx for idx, target in enumerate(test_targets) if target == mali_labels[1]]
        backdoored_data = [x for x in Subset(fl_test, indices)]
        for ind in range(len(backdoored_data)):
            x = add_trigger(backdoored_data[ind][0], pattern_size=3, rgb=[1, 1, 1], solid=False)
            backdoored_data[ind] = (x, backdoored_data[ind][1])
        other_indices = list(set(range(len(test_targets))) - set(indices))
        other_data = [x for x in Subset(fl_test, other_indices)]
        modified_test_set = ConcatDataset([backdoored_data, other_data])
        test_loader = DataLoader(modified_test_set, server_batch_size, shuffle=True, num_workers=5)

    # Set up containers
    model.eval()
    correct_pred = {class_id: 0 for class_id in range(class_num)}
    total_pred = {class_id: 0 for class_id in range(class_num)}
    per_class_accuracy = {}

    attack_success_rate = {}
    if len(mali_labels) != 0:
        if attack_method in ['label_perturbation', '2_label_flipping']:
            attack_success_rate = {class_id: 0 for class_id in mali_labels}
        elif attack_method in ['backdoor', 'dba', 'clear_label_backdoor']:
            attack_success_rate = {class_id: 0 for class_id in [mali_labels[0]]}

    # Evaluate on original test loader for accuracy
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in original_test_loader:
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
    test_loss /= len(original_test_loader)
    accuracy = correct / len(original_test_loader.dataset)
    for class_id in range(class_num):
        if total_pred[class_id] != 0:
            per_class_accuracy[class_id] = correct_pred[class_id] / total_pred[class_id]
        else:
            per_class_accuracy[class_id] = -1

    # Evaluate on modified loader for attack success rate
    if attack_success_rate:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True).flatten()
                target = target.flatten()
                for t, p in zip(target, pred):
                    if attack_method == '2_label_flipping':
                        if t == mali_labels[0] and p == mali_labels[1]:
                            attack_success_rate[mali_labels[0]] += 1
                        elif t == mali_labels[1] and p == mali_labels[0]:
                            attack_success_rate[mali_labels[1]] += 1
                    elif attack_method == 'backdoor':
                        if t == mali_labels[0] and p == mali_labels[1]:
                            attack_success_rate[mali_labels[0]] += 1
                    elif attack_method == 'dba':
                        if t != mali_labels[0] and p == mali_labels[0]:
                            attack_success_rate[mali_labels[0]] += 1
                    elif attack_method == 'clear_label_backdoor':
                        if t == mali_labels[1] and p == mali_labels[0]:
                            attack_success_rate[mali_labels[0]] += 1

        for class_id in attack_success_rate:
            if attack_method == 'dba':
                total_non_target = len(test_loader.dataset) - total_pred.get(mali_labels[0], 0)
                attack_success_rate[class_id] = (
                    attack_success_rate[class_id] / total_non_target if total_non_target > 0 else -1
                )
            else:
                attack_success_rate[class_id] = (
                    attack_success_rate[class_id] / total_pred[class_id] if total_pred[class_id] > 0 else -1
                )

    return test_loss, accuracy, per_class_accuracy, attack_success_rate


def grad_add_to_model(model, lr, grad):
    # update on the input model
    model_sd = model.state_dict()
    for key in model_sd.keys():
        if key not in grad:
            continue  # 🔁 Skip buffers like 'bn1.running_mean', etc.

        if isinstance(grad[key], torch.Tensor):
            grad_to_add = (lr * grad[key]).to(model_sd[key].dtype)
            model_sd[key] = (model_sd[key] + grad_to_add).to(model_sd[key].dtype)

    model.load_state_dict(model_sd)



def weight_to_model(model, weight):
    model_sd = model.state_dict()
    for key in model_sd.keys():
        model_sd[key] = weight[key]
    model.load_state_dict(model_sd)
    return model


def fltrust_server_iteration(iter, c_epochs, root_ds, mco_dict, batch_size):
        
    c_sd_before = copy.deepcopy(mco_dict[-1]["c_model"].state_dict())

    client_dl = DataLoader(root_ds, batch_size, shuffle=True, num_workers=0)
    # train the classifier model
    for epoch in range(c_epochs):
        loss, acc = train_c_model(train_loader = client_dl, 
                                  model = mco_dict[-1]["c_model"], 
                                  criterion = mco_dict[-1]["c_criterion"], 
                                  optimizer = mco_dict[-1]["c_optimizer"], 
                                  attack_method = 'none',
                                  client_id=-1,
                                  malicious_client_ids=[],
                                  optimizer_poison=None,
                                  dba_poisoning_per_batch=0
                                )
        if epoch % 1 == 0:
            logger.info(f"FLTrust server_ep{epoch}: C-Model, loss: {loss:.6f}, train acc: {acc:.6f}")
    
    
    c_grad = model_weight_diff(mco_dict[-1]["c_model"].state_dict(), c_sd_before)

    # remove the influance from the server training
    mco_dict[-1]["c_model"].load_state_dict(c_sd_before)

    return c_grad


def grad_weighted_sum(grad_dict, weights):

    new_grad_dict={x: 0 for x in grad_dict[list(grad_dict.keys())[0]].keys()}
    for client_id in grad_dict.keys():
        for layer, param in grad_dict[client_id].items():
            new_grad_dict[layer] += weights[client_id] * param
    return new_grad_dict



def mistakes_count(mali_labels, malicious_client_ids, clients_per_class_selection, attack_method):
    mistake_num = 0
    selection_num = 0
    if attack_method in ['backdoor', 'clear_label_backdoor']:
        mali_labels_ = [mali_labels[0]]
    else:
        mali_labels_ = mali_labels

    for class_id in mali_labels_:
        if class_id is not None:
            selections = clients_per_class_selection[class_id]
            mistakes = list(set(selections).intersection(malicious_client_ids))
            mistake_num += len(mistakes)
            selection_num += len(selections)
            if mistakes != []:
                logger.info(f"label {class_id} has {len(mistakes)} mistakes on client(s): {mistakes}")
    
    return selection_num, mistake_num


def weighted_avg_aggr(grad_dict, p_dict, device):
    new_grads = dict()
    client_ids = list(grad_dict.keys())
    # print("client_ids", client_ids)
    with torch.no_grad():
        for layer_name, param in grad_dict[client_ids[0]].items():
            # print("layer_name", layer_name)
            layer_weights = {}
            for client_id in client_ids:
                layer_weights[client_id] = grad_dict[client_id][layer_name]
            weighted_avg_w = layer_weighted_avg(layer_weights, p_dict)
            new_grads[layer_name] = weighted_avg_w.to(device)
            
    return new_grads


def layer_weighted_avg(layer_weights, p_dict):
    p_sum = sum(p_dict.values())
    # normalize p_dict
    for client_id, count in p_dict.items():
        p_dict[client_id] = count / p_sum
    
    layer_size = list(layer_weights.values())[0].size()
    if len(layer_size) != 0:
        avg_w = torch.zeros_like(list(layer_weights.values())[0])
        for client_id, weights in layer_weights.items():
            avg_w += weights.float() * p_dict[client_id]
    else:
        avg_w = list(layer_weights.values())[0]

    return avg_w


def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


class NegativeEntropyLoss(nn.Module):
    """
    A PyTorch loss that encourages output distributions with low confidence.
    This is typically used to generate adversarial or malicious updates.
    """
    def __init__(self):
        super(NegativeEntropyLoss, self).__init__()

    def forward(self, logits, target=None):  # Accept target to match PyTorch interface
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)  # shape: [batch_size]
        return -entropy.mean()