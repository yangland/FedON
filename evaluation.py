import clip
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import random
from data_process import resize_x_dataset
from utilities import wass_2_dists, crosseval_high_accuracy_subset, create_client_weights, selfeval_subset
import logging
from aggr.geometric_median import geometric_median
import math
import heapq
from collections import OrderedDict, Counter
from visualization import matrix_plot
from wassdistance import SinkhornDistance
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
from torchvision.utils import make_grid
from PIL import Image
import torch_fidelity
import subprocess
import re
import torch.nn.functional as F
from torch import nn
from sklearn.cluster import KMeans
import hdbscan

logger = logging.getLogger("individual_logging")
logger.addHandler(logging.StreamHandler())

def master_sensor_z_dist(master_sensor, train_dataset, device, sample_size, to_cpu=False):
    # convert dataset into the master sensor's latent z distribution
    if master_sensor == 'CLIP':
        CLIP, preprocess = clip.load("ViT-B/32", device=device)
    elif master_sensor == 'FID':
        inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(64)])
    else:
        return -1

    labels = list(set([int(x[1]) for x in train_dataset]))
    labels.sort()
    label_dist = {label: [] for label in labels}
    
    with torch.no_grad():
        for class_id in labels:
            indices = [idx for idx, target in enumerate(train_dataset) if target[1] == class_id]
            indices = random.sample(indices, sample_size)
            selected_dataset = Subset(train_dataset, indices)
            
            data_loader = torch.utils.data.DataLoader(selected_dataset,
                                                        batch_size=1,
                                                        shuffle=False)
            # print("data_loader length", len(data_loader))
            for x, y in data_loader:
                # denormalize x data back to [0, 255]
                # use the same function in torch.utils.save_image
                grid = make_grid(x)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                x_img = Image.fromarray(ndarr)                

                if master_sensor == 'CLIP':
                    # transform = T.ToPILImage()
                    # image = transform(x[0])
                    x_img = preprocess(x_img).unsqueeze(0).to(device)
                    output = CLIP.encode_image(x_img)[0] 
                    # print("clip output size", output.size())  # [512]
                elif master_sensor == 'FID':
                    # print(x.size())
                    # image = tensor_rgb_225(x, 1)
                    # print("image", image)
                    output = inception(x_img)[0]
                    # print("FID output size", output.size()) # [64]
                # print("output", output)
                label_dist[class_id].append(output)

    for class_id in labels:
        if to_cpu:
            label_dist[class_id] = torch.stack(label_dist[class_id]).detach().cpu()
        else:
            label_dist[class_id] = torch.stack(label_dist[class_id])

    return label_dist
   

def clients_cross_eval(client_selection, mco_dict, class_num, clients_g_imgs, clients_eval_mode, device):
    eval_g_dict = OrderedDict({key: {} for key in list(range(class_num))})
    eval_c_dict = OrderedDict({key: {} for key in list(range(class_num))})
    
    # if abbs:
    #     print(f"ablation model for FedEval: {abbs}")
    
    if clients_eval_mode == "cross-eval": 
        num_clients = len(client_selection)
        # Create mapping from client ID to matrix index
        client_to_idx = {client_id: idx for idx, client_id in enumerate(client_selection)}
        
        # Initialize matrices
        acc_M = [[0 for _ in range(num_clients)] for _ in range(num_clients)]
        per_class_acc_M = [[0 for _ in range(num_clients)] for _ in range(num_clients)]
        loss_M = [[0 for _ in range(num_clients)] for _ in range(num_clients)]
        acc_M_min = [[0 for _ in range(num_clients)] for _ in range(num_clients)]
        self_eval_M = {}
        client_preds = {}
        
        for client_i in client_selection:
            all_preds = []
            i = client_to_idx[client_i]
            
            for client_j in client_selection:
                j = client_to_idx[client_j]
                
                g_samples = clients_g_imgs[client_i]
                c_model = mco_dict[client_j]["c_model"]
                c_criterion = nn.CrossEntropyLoss()
                acc, per_class_acc, test_loss, predictions, targets = c_model_eval(
                    g_samples, c_model, c_criterion, class_num, device)
                
                acc_M[i][j] = acc
                per_class_acc_M[i][j] = per_class_acc
                acc_M_min[i][j] = min(per_class_acc.values()) + 1e-9
                loss_M[i][j] = test_loss

                if client_i == client_j:
                    self_eval_M[client_i] = per_class_acc
                
                all_preds.append(torch.tensor(predictions))
                
            preds_stack = torch.stack(all_preds) 
            client_preds[client_i] = torch.mean(preds_stack.double(), 0).long()

        R_avg = np.mean(np.array(acc_M))
        R_med = np.median(np.array(acc_M))
        logger.info(f"R_avg acc: {R_avg}")
        logger.info(f"R_med acc: {R_med}")
        logger.info(f"acc_M, {acc_M}")
        logger.info(f"per_class_acc_M, {per_class_acc_M}")
        logger.info(f"acc_M_min, {acc_M_min}")
        logger.info(f"loss_M, {loss_M}")

        # Evaluate C models
        for client_j in client_selection:
            j = client_to_idx[client_j]
            per_class_dict = {x: 0 for x in range(class_num)}
            
            for client_k in client_selection:
                k = client_to_idx[client_k]
                if k != j:  # only take other clients' evaluations
                    per_class_acc = per_class_acc_M[j][k]
                    for class_id, acc in per_class_acc.items():
                        per_class_dict[class_id] += acc
            
            per_class_dict = {k: v / (num_clients - 1) for k, v in per_class_dict.items()}    
            for class_id in range(class_num):
                eval_c_dict[class_id][client_j] = per_class_dict[class_id]
        
        logger.info(f"eval_c_dict, {eval_c_dict}")
        
        # Evaluate G models
        for client_k in client_selection:
            k = client_to_idx[client_k]
            per_class_dict = {x: 0 for x in range(class_num)}
            
            for client_j in client_selection:
                j = client_to_idx[client_j]
                if j != k:  # only take other clients' evaluations
                    per_class_acc = per_class_acc_M[j][k]
                    for class_id, acc in per_class_acc.items():
                        per_class_dict[class_id] += acc
            
            per_class_dict = {k: v / (num_clients - 1) for k, v in per_class_dict.items()}    
            for class_id in range(class_num):
                eval_g_dict[class_id][client_k] = per_class_dict[class_id]
        
        logger.info(f"eval_g_dict, {eval_g_dict}")
        
        # Self evaluation
        self_acc = {}
        for client_j in client_selection:
            j = client_to_idx[client_j]
            self_acc[client_j] = per_class_acc_M[j][j]
        
        avg_r_c, avg_r_g, avg_r_cg, min_r_c, min_r_g = consoliated_cross_eval(
            eval_g_dict, eval_c_dict, client_selection, class_num)    

        # Process test loss in loss_M
        loss_M = np.array(loss_M)
        acc_M = np.array(acc_M)
        L_avg = np.mean(loss_M)
        L_med = np.median(loss_M)

    elif clients_eval_mode == "no-eval":
        for client_id in client_selection:
            for class_id in range(class_num):
                eval_g_dict[class_id][client_id] = 0 
                eval_c_dict[class_id][client_id] = 0 
        # Initialize empty matrices for no-eval case
        acc_M = np.zeros((len(client_selection), len(client_selection)))
        per_class_acc_M = [[{} for _ in client_selection] for _ in client_selection]
        loss_M = np.zeros((len(client_selection), len(client_selection)))
        acc_M_min = np.zeros((len(client_selection), len(client_selection)))
        R_avg = R_med = L_avg = L_med = 0
        avg_r_c = avg_r_g = avg_r_cg = min_r_c = min_r_g = 0
        self_acc = {client_id: {} for client_id in client_selection}

    return eval_g_dict, eval_c_dict, acc_M, per_class_acc_M, \
            R_avg, R_med, avg_r_c, avg_r_g, avg_r_cg, self_acc, L_avg, L_med, loss_M, acc_M_min


def c_model_eval(test_dataset, c_model, criterion, class_num, device):
    class_list = list(range(class_num))
    samples_each_class = dict.fromkeys(class_list, 0)

    correct = 0
    correct_rev = 0

    test_loss = 0
    test_loss_rev = 0
    
    pre_class_correct = dict.fromkeys(class_list, 0)
    # pre_class_correct_rev = dict.fromkeys(class_list, 0)

    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = c_model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            correct_rev += target.eq(prediction.view_as(target)).sum().item()
            test_loss += criterion(output, target).item()
            predictions += torch.squeeze(prediction).tolist()
            targets += target.tolist()
            # print("output", output)
            # print("target", target)

            # reversed loss, label as output, prediction as target
            # one_hot_target = F.one_hot(target, num_classes=class_num).double().to(device)
            # print("one_hot_target", one_hot_target)
            # print("prediction", prediction)
            # print("prediction, squeeze", torch.squeeze(prediction))
            # test_loss_rev += criterion(one_hot_target, torch.squeeze(prediction)).item()
            
            for t_class in class_list:
                samples_each_class[t_class] += (target == t_class).sum().item()
                
                pre_class_correct[t_class] += \
                    (torch.flatten(prediction.eq(target.view_as(prediction))) * (target == t_class)).sum().item()
    
    acc_all = correct / len(test_loader.dataset)
    test_loss /= len(test_loader)
    test_loss_rev /= len(test_loader)
    for key, value in pre_class_correct.items():
        pre_class_correct[key] = value / samples_each_class[key]
    
    return acc_all, pre_class_correct, test_loss, predictions, targets


def g_model_generate_img(ddpm, t_class, num_samples, ws_test, image_size, device):
    ddpm.eval()
    size = tuple(image_size)
    
    with torch.no_grad():
        # Use inference mode for potentially better performance
        with torch.inference_mode():
            x_gen, _ = ddpm.sample_one_class(num_samples, t_class, size, guide_w=ws_test)
    return x_gen



def g_model_generate_dataset(g_model, class_list, samples_each_class, ws_test, image_size, device):
    g_model.eval()
    
    # Calculate batch sizes more efficiently
    batch_size = 100  # Optimal batch size may vary based on your GPU
    full_batches = samples_each_class // batch_size
    remainder = samples_each_class % batch_size
    
    # Pre-allocate lists
    X_list = []
    y_list = []
    
    # Generate samples in parallel for all classes
    for t_class in class_list:
        # Process full batches
        for _ in range(full_batches):
            x_gen = g_model_generate_img(g_model, t_class, batch_size, ws_test, image_size, device)
            X_list.append(x_gen)
            y_list.append(torch.full((batch_size,), t_class, device=device))
        
        # Process remainder
        if remainder > 0:
            x_gen = g_model_generate_img(g_model, t_class, remainder, ws_test, image_size, device)
            X_list.append(x_gen)
            y_list.append(torch.full((remainder,), t_class, device=device))
    
    # Concatenate all at once
    X = torch.cat(X_list)
    y = torch.cat(y_list)
    
    return TensorDataset(X, y)



def clients_g_model_generation(mco_dict, client_selection, class_num, sample_size, g_ws, image_size, device):
    class_list = list(range(class_num))
    client_g_imgs = {}

    # print("client_selection", client_selection)
    # print("mco_dict keys", mco_dict.keys())
    for client_id in client_selection:
        g_model = mco_dict[client_id]["g_model"]
        test_dataset = g_model_generate_dataset(g_model, class_list, sample_size, g_ws, image_size, device)
        client_g_imgs[client_id] = test_dataset
    return client_g_imgs


def g_c_model_eval(g_model, c_model, class_num, samples_each_class, ws_test, img_shape, device):
    g_model.eval()
    c_model.eval()
    correct = 0
    pre_class_correct = dict.fromkeys(list(range(class_num)), 0)
    
    X = []
    y = []
    
    # generate samples from the G model
    for t_class in range(class_num):
        x_gen = g_model_generate_img(g_model, t_class, samples_each_class, ws_test, img_shape)
        y_gen = torch.tensor([t_class] * samples_each_class)
        X.append(x_gen)
        y.append(y_gen)
    
    X = torch.stack(X)
    y = torch.stack(y)
    
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)
    
    # feed the generated data to the C model
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = c_model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            
            for t_class in range(class_num):
                pre_class_correct[t_class] += \
                    (prediction.eq(target.view_as(prediction)) * (target == t_class)).sum().item()
    
    acc_all = correct / len(test_loader.dataset)
    
    for key, value in pre_class_correct.items():
        pre_class_correct[key] = value / samples_each_class
    
    return acc_all, pre_class_correct

def self_eval_single_client(mco_dict, client_id, class_num, samples_each_class, ws_test, img_shape):
    g_model = mco_dict[client_id]["g_model"]
    c_model = mco_dict[client_id]["c_model"]
    acc_all, pre_class_correct = g_c_model_eval(g_model, c_model, class_num, samples_each_class, ws_test, img_shape)
    return acc_all, pre_class_correct


def master_sensor_eval(client_g_imgs, sensor_name, g_sample_each_class, device):
    latent_dict = {}
    
    for client_id, test_dataset in client_g_imgs.items():
        label_dist = master_sensor_z_dist(sensor_name, test_dataset, device, g_sample_each_class)
        latent_dict[client_id] = label_dist

    return latent_dict

def candidate_sensor_eval(client_g_imgs, unet_classes, candidate_sensor, image_size, device):
    latent_dict = {}
    change_to_rgb = True if image_size[0]==1 else False
    
    with torch.no_grad():
        for client_id, test_dataset in client_g_imgs.items():
            latent_dict[client_id] = {}
            for class_id in range(unet_classes):
                test_dataset_targets = [row[1] for row in test_dataset]
                indices = [idx for idx, target in enumerate(test_dataset_targets) if target == class_id]
                selected_dataset = Subset(test_dataset, indices)
                x_gen = [row[0] for row in selected_dataset]
                x_gen64 = resize_x_dataset(x_gen, 224, change_to_rgb)
            
                _, z_output, _, _ = candidate_sensor.vae(x_gen64.to(device))
                
                latent_dict[client_id][class_id] = z_output
    
    return latent_dict


def candidate_sensor_group_eval(candidate_sensors, unet_classes, client_g_imgs, image_size, device):
    candidate_sensors_z_dict ={}
    
    for s in candidate_sensors:
        candidate_sensor_z_dict = candidate_sensor_eval(client_g_imgs, unet_classes, s, image_size, device)
        candidate_sensors_z_dict[s.name] = candidate_sensor_z_dict
        
    return candidate_sensors_z_dict


def sensor_analysis(master_sensor_z_dict, candidate_sensors, candidate_sensors_z_dict,
                      eval_g_dict, eval_c_dict, class_num, client_selection, device, iter, exp_dir, sensor_measure):
    sensor_weights = []
    
    if_master_sensor = True
    if_candidate_sensor = True
    multi_candidate_sensors = True
    
    if master_sensor_z_dict is None and  candidate_sensors_z_dict is None:
        logger.info(f"No sensor to work with")
        if_master_sensor = False
        if_candidate_sensor = False
        multi_candidate_sensors = False
    elif master_sensor_z_dict is None and  candidate_sensors_z_dict is not None:
        if_master_sensor=False
        if len(candidate_sensors) ==1:
            multi_candidate_sensors = False
    elif master_sensor_z_dict is not None and  candidate_sensors_z_dict is None:
        if_candidate_sensor = False
        multi_candidate_sensors = False
    
    logger.info(f"if_master_sensor: {if_master_sensor}")
    logger.info(f"if_candidate_sensor: {if_candidate_sensor}")
    logger.info(f"multi_candidate_sensors: {multi_candidate_sensors}")
    
    if if_candidate_sensor:
        if multi_candidate_sensors:
            # selecte the related candidate sensor Sr
            sr_index, related_candidate_measure = \
                get_most_relevant_candidate_sensor(client_selection, candidate_sensors, master_sensor_z_dict, device)
            logger.info(f"The most related sensor: {candidate_sensors[sr_index].name}")
        else:
            sr_index = 0
            related_candidate_measure = dict()
            logger.info(f"There is only 1 candidate sensor: {candidate_sensors[sr_index].name}")
        
        sr_z_dict = candidate_sensors_z_dict[candidate_sensors[sr_index].name]    
        sr_weights = client_per_class_aggr(sr_z_dict, client_selection, aggr=sensor_measure, device=device)
        
        for class_id, per_class_weight in sr_weights.items():
            logger.info(f"class_id {class_id} weight: {per_class_weight}")
    else:
        sr_index, related_candidate_measure = None, dict()
        sr_weights = dict()
        logger.info(f"There is NO candidate sensor")                 
    
    if if_master_sensor and if_candidate_sensor:
        w_sm, w_sr, sm_label_dict, sr_label_dict, avg_resos = \
            weight_distr_bet_master_candidate_sensors(client_selection, sr_z_dict, master_sensor_z_dict,
                                                       class_num, device, True, iter, exp_dir)
        logger.info(f"Weights between master and selected sensor: {w_sm:.4f}, {w_sr:.4f}")
    elif if_master_sensor and if_candidate_sensor==False:
        logger.info(f"There is only master sensor")
        w_sm, w_sr = 1, 0
        avg_resos = []
    elif if_master_sensor==False and if_candidate_sensor:
        w_sm, w_sr = 0, 1
        avg_resos = []
    else:
        w_sm, w_sr = 0, 0
        avg_resos = []        
    w_sm_sr = [w_sm, w_sr]
    logger.info(f"w_sm, w_sr: {w_sm, w_sr}")
       
    if if_master_sensor:      
        sm_weights = client_per_class_aggr(master_sensor_z_dict, client_selection, aggr=sensor_measure, device=device)
        logger.info(f"sm_weights {w_sm:.4f}:")
        for class_id, per_class_weight in sm_weights.items():
            logger.info(f"class_id {class_id} weight: {per_class_weight}")
    else:
        sm_weights = dict()
    
    sensor_weights = consolidated_sensor_weights(w_sm, w_sr, sm_weights, sr_weights, client_selection, class_num)
    if if_master_sensor or if_candidate_sensor:
        logger.info(f"sensor_weights:")
        for class_id, per_class_weight in sensor_weights.items():
            logger.info(f"class_id {class_id} sensor weight: {per_class_weight}")
    
    return sr_index, related_candidate_measure, sm_weights, sr_weights, w_sm_sr, sensor_weights, avg_resos


def select_client_class_rep(client_selection, fl_class_num, clients_weights):
    # For each class label, only select the closest (client_num/2 -1) clients
    # these clients are allowed to generated the dataset in their G-model for server model training
    client_num_select = math.floor(len(client_selection)/2)
    # client_num_select = 1
    clients_per_class_selection = {}
    for class_id in range(fl_class_num):
        sorted_clients = {k: v for k, v in sorted(clients_weights[class_id].items(), 
                                                  key=lambda item: item[1], 
                                                  reverse = True)}
        
        selected_client_ids = list(sorted_clients.keys())[:client_num_select]
        clients_per_class_selection[class_id] = selected_client_ids
    # print("client_selection", clients_per_class_selection)
    return clients_per_class_selection

def client_per_class_aggr(z_dict, client_selection, aggr, device):
    # aggr in "geometric_median", "krum", "coordinate_wise_median"
    # client_num = len(z_dict)
    # print("z_dict i size", z_dict[0][0].size()) # [20, 512]
    # print("z_dict keys", list(z_dict.keys())) # 0, 1, 2, 3
    # print("z_dict[0][0]", z_dict[0][0])
    # print("z_dict[1][0]", z_dict[1][0])
    # print("z_dict[2][0]", z_dict[2][0])

    class_num = len(list(z_dict.values())[0])
    
    class_dict = {key: {} for key in range(class_num)}
    class_weights = {key: {} for key in range(class_num)}  

    # print("client_selection", client_selection)

    for client_id in client_selection:
        for class_id in range(class_num):
            class_dict[class_id][client_id] = z_dict[client_id][class_id]
    
    for class_id in range(class_num):
        if aggr=="geometric_median":
            _, per_class_weight = geometric_median(class_dict[class_id])
            # print(class_id, per_class_weight)
            class_weights[class_id] = weights_normalize(per_class_weight)
        if aggr=="Wasserstein":
            sinkhorn = SinkhornDistance(eps=0.001, max_iter=500, device=device)
            matrix = []
            for client_i in client_selection:
                wass_sum = 0
                row = []
                for client_j in client_selection:
                    a_z = class_dict[class_id][client_i].to(device)
                    b_z = class_dict[class_id][client_j].to(device)

                    wass, P, C = sinkhorn(a_z, b_z)
                    row.append(wass.item())
                    wass_sum+=wass
                # print("client_i, wass_sum", client_i, wass_sum)
                matrix.append(row)
                class_weights[class_id][client_i] = float(1 / (1e-9 + wass_sum))
            # print("matrix class id:", class_id)
            # print( np.matrix(matrix))
            
            class_weights[class_id] = weights_normalize(class_weights[class_id])
   

    return class_weights

def weights_normalize(client_weights):
    # total_w = sum(client_weights.values())
    max_w = max(client_weights.values())
    for client_id, weight in client_weights.items():
        client_weights[client_id] = weight/max_w * 100
    return client_weights

def get_most_relevant_candidate_sensor(client_selection, candidate_sensors, master_sensor_z_dict, device):
    most_relevant = []
    for client_id in client_selection:
        dist_means = []
        for i in range(len(candidate_sensors)):
            mean_, min_, matrix = distance_two_dists(candidate_sensors[i].master_view_dist, 
                                                    master_sensor_z_dict[client_id],
                                                    device)
            dist_means.append(mean_)
        min_mean = min(dist_means)
        most_relevant.append(dist_means.index(min_mean))
    
    measure = OrderedDict()
    for i in range(len(candidate_sensors)):
        measure[i] = Counter(most_relevant)[i]

    return most_frequent(most_relevant), measure


def most_frequent(List):
    return max(set(List), key = List.count)


def distance_two_dists(dist1, dist2, device):
    matrix = wass_2_dists(dist1, dist2, device)
    mean = np.mean(np.array(matrix))
    min = np.min(np.array(matrix))
    return mean, min, matrix


def weight_distr_bet_master_candidate_sensors(client_selection, sr_z_dict, m_z_dict, class_num, device, if_plot, iter, exp_dir):
    sm_label_dict = {}
    sr_label_dict = {}
    sm_reso = []
    sr_reso = []
    base_sm = np.zeros((class_num, class_num))
    base_sr = np.zeros((class_num, class_num))
    
    for i in client_selection:
        sm_label_dict[i] = {}
        sr_label_dict[i] = {}
        
        for class_id in range(class_num):
            sm_label_dict[i][class_id]=[]
            sr_label_dict[i][class_id]=[]
        
        for j in client_selection:
            if i != j:
                _, _, sm_matrix = distance_two_dists(m_z_dict[i], m_z_dict[j], device)
                _, _, sr_matrix = distance_two_dists(sr_z_dict[i], sr_z_dict[j], device)

                base_sm+=sm_matrix
                base_sr+=sr_matrix

                for class_id in range(class_num):
                    sm_label_dict[i][class_id].append(sm_matrix.diagonal()[class_id])
                    sr_label_dict[i][class_id].append(sr_matrix.diagonal()[class_id])
                    
                sm_reso.append(reso_score(sm_matrix))
                sr_reso.append(reso_score(sr_matrix))
    
    base_sm /= len(client_selection) * len(client_selection) 
    base_sr /= len(client_selection) * len(client_selection) 
    
    if if_plot:
        sm_title = f"Master sensor, iter {iter}"
        matrix_plot(base_sm, sm_title, f"S_m_{iter}", "class_id", "class_id", f"{exp_dir}/plots")
        sr_title = f"Most relevant sensor, iter {iter}"
        matrix_plot(base_sr, sr_title, f"S_r_{iter}", "class_id", "class_id", f"{exp_dir}/plots")
    
    avg_sm_reso = np.mean(sm_reso)
    avg_sr_reso = np.mean(sr_reso)

    w_sm = avg_sm_reso/(avg_sm_reso + avg_sr_reso)
    w_sr = avg_sr_reso/(avg_sm_reso + avg_sr_reso)
    
    return w_sm, w_sr, sm_label_dict, sr_label_dict, (avg_sr_reso, avg_sm_reso)


def consolidated_sensor_weights(w_sm, w_sr, sm_weights, sr_weights, 
                                client_selection, class_num):
    sensor_weights = {}
    for class_id in range(class_num):
        if w_sm!=0:
            sm_class_w = sm_weights[class_id].values()
        else:
            sm_class_w = dict.fromkeys(client_selection, 0)
        
        if w_sr!=0:
            sr_class_w = sr_weights[class_id].values()
        else:
            sr_class_w = dict.fromkeys(client_selection, 0)        

        sm_class_w = [x * w_sm for x in sm_class_w]
        sr_class_w = [x * w_sr for x in sr_class_w]
        sensor_weights_list = [sum(x) for x in zip(sm_class_w, sr_class_w)]
        sensor_weights[class_id] = dict(zip(client_selection, sensor_weights_list))

    return sensor_weights


def consoliated_cross_eval(eval_g_dict, eval_c_dict, client_selection, class_num):
    # select based on both C and G eval 
    # minmax as selection
    # eval_c_dict[class_id][client_selection[j]] = per_class_dict[class_id] # value scale
    c_avg = {}
    g_avg = {}
    cg_avg = {}
    c_min = {}
    g_min = {}
    
    client_acc_c = {}
    client_acc_g = {}
    
    for client_id in client_selection:
        client_acc_c[client_id] = []
        client_acc_g[client_id] = []
        for class_id in range(class_num):
            client_acc_c[client_id].append(eval_c_dict[class_id][client_id])
            client_acc_g[client_id].append(eval_g_dict[class_id][client_id])
        c_avg[client_id] = np.mean(client_acc_c[client_id])
        g_avg[client_id] = np.mean(client_acc_g[client_id])
        cg_avg[client_id] = c_avg[client_id] + g_avg[client_id]
        c_min[client_id] = np.min(client_acc_c[client_id])
        g_min[client_id] = np.min(client_acc_g[client_id])
        
    c_avg = sorted(c_avg.items(), key=lambda x: x[1], reverse=True)
    g_avg = sorted(g_avg.items(), key=lambda x: x[1], reverse=True)
    cg_avg = sorted(cg_avg.items(), key=lambda x: x[1], reverse=True)
    c_min = sorted(c_min.items(), key=lambda x: x[1], reverse=True)
    g_min = sorted(g_min.items(), key=lambda x: x[1], reverse=True)
        
    return c_avg, g_avg, cg_avg, c_min, g_min


def get_consolidated_client_weights(w_sm, w_sr, sm_weights, sr_weights, 
                                    client_selection, eval_g_dict, eval_c_dict, class_num, type):
    sensor_weights = {}
    multi_dict = {}
  
    eval_c_avg = {}
    eval_g_avg = {}
    for client_id in client_selection:
        eval_c_avg[client_id] = 0
        eval_g_avg[client_id] = 0
        for class_id in range(class_num):
            eval_c_avg[client_id] += eval_c_dict[class_id][client_id]
            eval_g_avg[client_id] += eval_g_dict[class_id][client_id]
        eval_c_avg[client_id] = eval_c_avg[client_id]/class_num
        eval_g_avg[client_id] = eval_g_avg[client_id]/class_num 
  
  
    if w_sm != 0 or w_sr!=0:
        for class_id in range(class_num):
            if w_sm!=0:
                sm_class_w = sm_weights[class_id].values()
            else:
                sm_class_w = dict.fromkeys(client_selection, 0)
            
            if w_sr!=0:
                sr_class_w = sr_weights[class_id].values()
            else:
                sr_class_w = dict.fromkeys(client_selection, 0)        

            sm_class_w = [x * w_sm for x in sm_class_w]
            sr_class_w = [x * w_sr for x in sr_class_w]
            sensor_weights_list = [sum(x) for x in zip(sm_class_w, sr_class_w)]
            sensor_weights[class_id] = dict(zip(client_selection, sensor_weights_list))
    else:
        for class_id in range(class_num):
            sensor_weights[class_id] = dict.fromkeys(client_selection, 1)
    
    if type == "cross_eval1":
        logger.info(f"Non sensors, use only clients eval results")
        sensor_weights = eval_g_dict  
    if type == "sensor_only":
        logger.info(f"only sensor")

    elif type == "cross_eval2":
        # select based on both C and G eval 
        # minmax as selection
        c_min = {}
        g_min = {}
        cg_min = {}
        for client_id in client_selection:
            min_c_acc = min(eval_c_dict[class_id])
            min_g_acc = min(eval_g_dict[class_id])
            c_min[client_id] = min_c_acc
            g_min[client_id] = min_g_acc
            cg_min[client_id] = min_c_acc + min_g_acc
        c_min = sorted(c_min.items(), key=lambda x: x[1], reverse=True)
        g_min = sorted(g_min.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"c_min: {c_min}")
        logger.info(f"g_min: {g_min}")
        
        sensor_weights = cg_min
        #TODO sensor_weight should be class level not client level
        #TODO rule based selection
        
    elif type == "sensor_g_cross_eval":
        for class_id in range(class_num):
            multi_dict[class_id] = {}
            for client_id in client_selection:
                # clients_weights[class_id][client_id] *= math.exp(clients_eval_dict[class_id][client_id])
                # print("R", 1 + eval_c_avg[client_id])
                # print("r", 1 + eval_g_dict[class_id][client_id])
                clients_eval_multi = math.pow(1 + eval_c_avg[client_id],  1 + eval_g_dict[class_id][client_id])
                sensor_weights[class_id][client_id] *= clients_eval_multi
                sensor_weights[class_id][client_id] = float(sensor_weights[class_id][client_id])
                multi_dict[class_id][client_id] = clients_eval_multi

        logger.info(f"multi_dict:")
        for class_id, multi in multi_dict.items():
            logger.info(f"class_id {class_id} multi: {multi}")

    return sensor_weights, multi_dict

def reso_score(matrix):
    m = np.matrix(matrix)
    size = m.shape[0]
    trace = np.trace(m)
    score = 1 - (trace/size) / (m.sum()/(size * size))
    return score


def fid_bet_ds(ds_1, ds_2, channels, class_num):
    fid_per_class = OrderedDict()
    for class_id in range(class_num):
        ind1 = [idx for idx, target in enumerate(ds_1) if ds_1[idx][1] == class_id]
        ind2 = [idx for idx, target in enumerate(ds_2) if ds_2[idx][1] == class_id]
        ds1 = [ds_1[i][0] for i in ind1]
        ds2 = [ds_2[i][0] for i in ind2]
        ds1 = torch.stack(ds1)
        ds2 = torch.stack(ds2)
        fid_score = fid_two_tensors(ds1, ds2, channels)
        fid_per_class[class_id] = fid_score
    avg_fid = np.mean(list(fid_per_class.values()))
    return fid_per_class, avg_fid


def fid_two_tensors(ds_1, ds_2, channels):
    # ds_1 = tensor_rgb_225(ds_1, channels)
    # ds_2 = tensor_rgb_225(ds_2, channels)
    # ds_1 = ds_1.to(torch.uint8)
    # ds_2 = ds_2.to(torch.uint8)

    ds_1 = tensor_image_denormalization(ds_1)
    ds_2 = tensor_image_denormalization(ds_2)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(ds_1, real=True)
    fid.update(ds_2, real=False)
    fid_score = fid.compute().item()
    return fid_score


def tensor_rgb_225(ds, channels):
    new_ds = []
    transform = T.Compose([T.ToPILImage(), 
                    T.PILToTensor()])
    for i in range(ds.size()[0]):
        x = transform(ds[i])
        if channels == 1:
            x = x.repeat( 3, 1, 1)
        new_ds.append(x)
    return torch.stack(new_ds)


# def tensor_image_denormalization(ds):
#     image_t = []
#     transform = T.Compose([T.PILToTensor()])
#     for x in ds:
#         # denormalize x data back to [0, 255]
#         # use the same function in torch.utils.save_image
#         grid = make_grid(x)
#         ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
#         x_img = Image.fromarray(ndarr) 
#         image_t.append(transform(x_img))
#     return torch.stack(image_t)



def tensor_image_denormalization(ds):
    if ds is None:
        raise ValueError("Input dataset cannot be None")
    
    if len(ds) == 0:
        return torch.Tensor()  # Return empty tensor for empty input
    
    image_t = []
    transform = T.Compose([T.PILToTensor()])
    
    for i, x in enumerate(ds):
        try:
            # Check if tensor is valid
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Element {i} is not a tensor, got {type(x)}")
                
            if x.dim() not in [3, 4]:
                raise ValueError(f"Tensor {i} has invalid shape {x.shape}. Expected 3D or 4D tensor")
            
            # Process tensor
            grid = make_grid(x)  # Handles batch dimension if present
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            
            # Convert to PIL Image
            x_img = Image.fromarray(ndarr)
            
            # Transform back to tensor
            image_t.append(transform(x_img))
            
        except Exception as e:
            print(f"Error processing tensor {i}: {str(e)}")
            print(f"Tensor shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
            print(f"Tensor type: {type(x)}")
            raise  # Re-raise if you want to stop on error
    
    if not image_t:  # If all failed
        raise RuntimeError("All tensors failed processing")
    
    return torch.stack(image_t)


def fid_bet_folders(input1, input2, device):
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=input1,    # <<< directory 1
        input2=input2,    # <<< directory 2
        cuda= device, # True
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        samples_find_deep=True,
    )
    fid = metrics_dict['frechet_inception_distance']
    return fid


# def calc_fid(gpu, fid_subprocess: bool, input1, input2):
#     if fid_subprocess:
#         cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
#         cmd_arr = cmd.split(' ')
#         res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
#         output = str(res.stdout)
#         m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
#         fid = float(m.group(1))
#     else:
#         metrics_dict = torch_fidelity.calculate_metrics(
#             input1=input1,
#             input2=input2,
#             cuda=True,
#             isc=False,
#             fid=True,
#             kid=False,
#             verbose=False,
#             samples_find_deep=True,
#         )
#         fid = metrics_dict['frechet_inception_distance']
#     return fid


def calc_fid(gpu, fid_subprocess: bool, input1, input2):
    """More robust FID calculation with multiple fallbacks"""
    attempts = [
        {"method": "subprocess_gpu", "cuda": True},
        {"method": "direct_cpu", "cuda": False},
        {"method": "subprocess_cpu", "cuda": False}
    ]
    
    for attempt in attempts:
        try:
            if attempt["method"].startswith("subprocess"):
                gpu_arg = f"--gpu {gpu}" if attempt["cuda"] and gpu is not None else ""
                cmd = f"fidelity {gpu_arg} --fid --input1 {input1} --input2 {input2} --silent"
                res = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                output = res.stdout
                m = re.search(r'frechet_inception_distance:\s*(\d+\.\d+)', output)
                if m:
                    return float(m.group(1))
            else:
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=input1,
                    input2=input2,
                    cuda=attempt["cuda"],
                    isc=False,
                    fid=True,
                    kid=False,
                    verbose=False,
                )
                return metrics_dict['frechet_inception_distance']
        except Exception as e:
            print(f"Attempt {attempt['method']} failed: {str(e)}")
            continue
    
    print("All FID calculation methods failed")
    return float('inf')



def clients_weights_from_analysis(client_selection, sensor_weights, R_avg, R_med, avg_r_c, avg_r_g, avg_r_cg, \
    L_avg, L_med, loss_M, acc_M, acc_M_min, per_class_acc_M, select_beta, client_list, method, mode, exp_dir, \
        iter, k1, k2, abbs):
    
    if abbs:
        print(f"Ablation model for FedEval: {abbs}")
    
    # limited selection based on R_avg
    max_select_num = math.floor(len(client_selection) * select_beta)
    num_clients = len(client_selection)
    logger.info(f"client_selection: {client_selection}")
    logger.info(f"acc_M: {acc_M}")
    logger.info(f"loss_M: {loss_M}")
    logger.info(f"method: {method}")
    logger.info(f"MODE: {mode}")
    
    
    matrix_plot(acc_M, f"acc_M{iter}", f"acc_M_{iter}", "client_id", "client_id", f"{exp_dir}/plots", client_selection)
    matrix_plot(loss_M, f"loss_M{iter}", f"loss_M_{iter}", "client_id", "client_id", f"{exp_dir}/plots", client_selection)
    # matrix_plot(acc_M_min, f"acc_M_min{iter}", f"acc_M_min_{iter}", "client_id", "client_id", f"{exp_dir}/plots", client_selection)

    curr_select_num = max_select_num
    logger.info(f"curr_select_num , {curr_select_num}")
    benign_client_ids =[]
    
    if method == 'KMeans':
        if mode == 'loss':
            matrix = loss_M
        elif mode == 'acc':
            matrix = acc_M

        G_kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(matrix)
        C_kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(np.swapaxes(matrix,0,1))
        G_avg_dict = group_rank(matrix, G_kmeans.labels_)
        C_avg_dict = group_rank(np.swapaxes(matrix,0,1), C_kmeans.labels_)
        print("G_avg_dict", G_avg_dict)
        print("C_avg_dict", C_avg_dict)
        
        if mode == 'acc':
            G_main_label = max(G_avg_dict, key=G_avg_dict.get)
            C_main_label = max(C_avg_dict, key=C_avg_dict.get)
        elif mode == 'loss':
            G_main_label = min(G_avg_dict, key=G_avg_dict.get)
            C_main_label = min(C_avg_dict, key=C_avg_dict.get)        

        logger.info(f"G_kmeans labels: {G_kmeans.labels_}, select group: {G_main_label}")
        logger.info(f"C_kmeans labels: {C_kmeans.labels_}, select group: {C_main_label}")

        clients_weights_c = {}
        clients_weights_g = {}

        for i in range(len(client_selection)):
            client_id = client_selection[i]

            if G_kmeans.labels_[i] == G_main_label:
                if mode == 'acc':
                    clients_weights_g[client_id] = matrix.sum(axis=1)[i] 
                elif mode == 'loss':
                    clients_weights_g[client_id] = 1 / matrix.sum(axis=1)[i] * 100
            else:
                clients_weights_g[client_id] = 0

            if C_kmeans.labels_[i] == C_main_label:
                if mode == 'acc':
                    clients_weights_c[client_id] = matrix.sum(axis=0)[i] 
                elif mode == 'loss':
                    clients_weights_c[client_id] = 1 / matrix.sum(axis=0)[i]  * 100
            else:
                clients_weights_c[client_id] = 0
    elif method == 'med':
        if mode == 'loss':
            med_l_c = dict(zip(client_selection, np.median(loss_M, axis=0)))
            med_l_g = dict(zip(client_selection, np.median(loss_M, axis=1))) 
            print("med_l_c", med_l_c)
            print("med_l_g", med_l_g)
            med_l_c = sorted(med_l_c.items(), key=lambda x: x[1], reverse=False)
            med_l_g = sorted(med_l_g.items(), key=lambda x: x[1], reverse=False)
            print("med_l_c sorted", med_l_c)
            print("med_l_g sorted", med_l_g)
            c_source = med_l_c
            g_source = med_l_g

        elif mode == 'acc':
            med_r_c = dict(zip(client_selection, np.median(acc_M, axis=0)))
            med_r_g = dict(zip(client_selection, np.median(acc_M, axis=1))) 
            print("med_r_c", med_r_c)
            print("med_r_g", med_r_g)
            med_r_c = sorted(med_r_c.items(), key=lambda x: x[1], reverse=True)
            med_r_g = sorted(med_r_g.items(), key=lambda x: x[1], reverse=True)
            print("med_r_c sorted", med_r_c)
            print("med_r_g sorted", med_r_g)
            c_source = med_r_c
            g_source = med_r_g     
        
        c_select = [x[0] for x in c_source[:curr_select_num]]
        g_select = [x[0] for x in g_source[:curr_select_num]]
        
        c_g_select_intersect = list(set(c_select) & set(g_select))
        logger.info(f"c_select, {c_select}")
        logger.info(f"g_select, {g_select}")
        logger.info(f"c g intersection, {c_g_select_intersect}")

        c_source = dict(c_source)
        g_source = dict(g_source)

        if mode == 'loss':
            clients_weights_c = dict((k, 1/c_source[k]*100) for k in c_select)
            clients_weights_g = dict((k, 1/g_source[k]*100) for k in g_select)
        elif mode == 'acc':
            # clients_weights_c = dict((k, c_source[k]) for k in c_select)
            clients_weights_g = dict((k, g_source[k]) for k in g_select)
            clients_weights_c = clients_weights_g
            

        for client_id in client_list:
            if client_id not in clients_weights_c.keys():
                clients_weights_c[client_id] = 0
            if client_id not in clients_weights_g.keys():
                clients_weights_g[client_id] = 0
    elif method == 'med_self_acc':
        acc_M_no_diag = acc_M[~np.eye(acc_M.shape[0],dtype=bool)].reshape(acc_M.shape[0],-1)
        med_r_g = dict(zip(client_selection, np.median(acc_M_no_diag, axis=1))) 
        self_acc = dict(zip(client_selection, np.diag(acc_M)))
        clients_weights_g = {}

        for client_id in client_selection:
            clients_weights_g[client_id] = med_r_g[client_id] * self_acc[client_id]
        
        clients_weights_c = clients_weights_g
    elif method == 'min_med_self_acc':
        acc_M_no_diag = acc_M[~np.eye(acc_M.shape[0],dtype=bool)].reshape(acc_M.shape[0],-1)
        med_r_g = dict(zip(client_selection, np.median(acc_M_no_diag, axis=1))) 
        self_acc = dict(zip(client_selection, np.diag(acc_M)))
        clients_weights_g = {}

        for client_id in client_selection:
            clients_weights_g[client_id] = min(med_r_g[client_id], self_acc[client_id])
        
        clients_weights_c = clients_weights_g
    
    elif method == 'HDBSCAN':
        # using HDBSCAN to select clients with per_class_acc_M
        def flatten_dict(d):
            # Sort the dictionary keys to ensure consistent ordering
            return [d[key] for key in sorted(d.keys())]

        if isinstance(per_class_acc_M, torch.Tensor):
            per_class_acc_M = per_class_acc_M.detach().cpu().numpy() 
        if isinstance(per_class_acc_M, list):
            per_class_acc_M = np.array(per_class_acc_M)
        
        print("per_class_acc_M size", per_class_acc_M.shape)

        # Convert the nested structure to a 2D numpy array
        flattened_data = []
        for client_data in per_class_acc_M:
            client_flattened = []
            for d in client_data:
                client_flattened.extend(flatten_dict(d))  # Concatenate all dictionary values into a single array
            flattened_data.append(client_flattened)

        print("per_class_acc_flattened", flattened_data)
        
        # Fit HDBSCAN on the prepared data
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,
                                    min_samples=1,
                                    allow_single_cluster=True).fit(flattened_data)
        cross_benign_idx = []
        max_num_in_cluster=0
        max_cluster_index=0
        print("clusterer.labels_", clusterer.labels_)
        if clusterer.labels_.max() < 0:
            for i in range(num_clients):
                cross_benign_idx.append(i)
        else:
            for index_cluster in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    cross_benign_idx.append(i)
        benign_client_ids = [client_selection[i] for i in cross_benign_idx]
        logger.info(f"HDBSCAN benign_client_ids: {benign_client_ids}")
        logger.info(f"HDBSCAN mali_client_ids: {[x for x in client_selection if x not in benign_client_ids]}")

        clients_weights_c, clients_weights_g = create_client_weights(client_selection, benign_client_ids, client_list)
        logger.info(f"benign clients weights: {clients_weights_c}")
    elif method == 'acc_subset_max':
        # using acc_M to select clients, in greed algorithm
        accuracy_matrix = np.array(acc_M)
        
        # clients cross evaluation
        cross_benign_idx = crosseval_high_accuracy_subset(accuracy_matrix)
        cross_eval_benign_ids = [client_selection[i] for i in cross_benign_idx]
        logger.info(f"cross_eval_benign_ids: {cross_eval_benign_ids}")
        
        
        # clients self evaluation
        self_eval_benign_idx, mean, std_dev = selfeval_subset(accuracy_matrix, multi=3)
        self_eval_benign_ids = [client_selection[i] for i in self_eval_benign_idx]
        logger.info(f"self_eval_benign_ids: {self_eval_benign_ids} mean: {mean} std_dev: {std_dev}")
        
        if abbs is False:
            # Find intersection between both sets
            intersection = set(cross_eval_benign_ids).intersection(set(self_eval_benign_ids))
        elif abbs == "cross":
            print("clients selection based on cross eval")
            intersection = cross_eval_benign_ids
        elif abbs == "self":
            print("clients selection based on self eval")
            intersection = self_eval_benign_ids
        
        # If intersection exists, assign it to benign_client_ids
        if intersection:
            benign_client_ids = sorted(list(intersection))
        else:
            # Otherwise, assign cross_eval_benign_ids
            benign_client_ids = cross_eval_benign_ids
        
        
        logger.info(f"acc_subset_max benign_client_ids: {benign_client_ids}")
        logger.info(f"acc_subset_max mali_client_ids: {[x for x in client_selection if x not in benign_client_ids]}")        
        
        clients_weights_c, clients_weights_g = create_client_weights(client_selection, benign_client_ids, client_list)
            
    elif method == 'min_med_self_acc_pow2':
        acc_M_no_diag = acc_M[~np.eye(acc_M.shape[0],dtype=bool)].reshape(acc_M.shape[0],-1)
        med_r_g = dict(zip(client_selection, np.median(acc_M_no_diag, axis=1))) 
        
        acc_M_t = acc_M.transpose()
        acc_M_t_no_diag = acc_M_t[~np.eye(acc_M_t.shape[0],dtype=bool)].reshape(acc_M_t.shape[0],-1)
        med_r_c = dict(zip(client_selection, np.median(acc_M_t_no_diag, axis=1))) 
        
        self_acc = dict(zip(client_selection, np.diag(acc_M)))
        clients_weights_g = {}

        for client_id in client_selection:
            # clients_weights_g[client_id] = pow(min(med_r_g[client_id], self_acc[client_id]), 2)
            clients_weights_g[client_id] = min(med_r_g[client_id], med_r_c[client_id], self_acc[client_id])
        
        logger.info(f"org clients_weights_g: {clients_weights_g}")

        std = np.std(list(clients_weights_g.values()))
        
        # k = 4
        # p = 2*(1-q)*std + q
        # p = (2 - 2*k)/(R_med+1) + 2*k -1
        p = 2*(k1 - k2)/(R_med+1) + (2*k2 - k1)

        logger.info(f"start power k1: {k1}, max power k2: {k2}, power for current iter {p}, r_med: {R_med}, std: {std}")

        for client_id in client_selection:    
            clients_weights_g[client_id] = pow(clients_weights_g[client_id], p)
        
        clients_weights_g = weights_normalize(clients_weights_g)
        clients_weights_c = clients_weights_g

    return clients_weights_c, clients_weights_g, benign_client_ids


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def group_rank(matrix, labels):
    summ = matrix.sum(axis=1)
    avg_dict = {}
    for label in list(set(labels)):
        idx = [idx for idx, target in enumerate(labels) if target==label]
        avg_dict[label] = np.mean([summ[index] for index in idx])
    return avg_dict
        
