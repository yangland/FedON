import datetime
import logging
import os
import random
import time
from collections import Counter
from logging import FileHandler, Formatter
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score

from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision.transforms as T
from tqdm import tqdm
from collections import defaultdict
from utilities import (
    create_client_weights
)
from helper import *
from csv_logging import CsvLogging
from data_process import create_fl_trust_root_dataset

from aggr.flame import flame_aggr
from aggr.fltrust import cosScoreAndClipValue
from aggr.fedavg_median import coordinate_wise_median, fedavg, trimmed_mean
from aggr.geometric_median import geo_med_aggr
from aggr.krum import krum_aggr
from aggr.normbound import normbound_aggr
from FedOda import *

logger = logging.getLogger("individual_logging")

def fl_training(fl_train, fl_test, params, image_size, exp_dir, timestamp):
    client_num = params['n']
    malicious_num = params['m']
    global device
    device  = params['cuda_device']
    noniid_type = params['noniid_type']
    
    global label_skew_alpha
    try:
        label_skew_alpha = params['label_skew_alpha']
    except:
        label_skew_alpha = None
        
    global p_degree  
    try:
        p_degree = params['p-degree']
    except:
        p_degree = None
    try:
        dirichlet_alpha = params['dirichlet_alpha']
    except:
        dirichlet_alpha = None
        
        
    attack_method = params['attack_method']
    
    if malicious_num == 0 or attack_method == 'none':
        if_attack = False
    else:
        if_attack = True
    
    defense = params['defense']
    aggr = params['aggr']
    aggr_c_lr = params['aggr_c_lr']
    
    global fl_class_num
    fl_class_num = params['fl_class_num']
    fl_iter_num = params['fl_iter_num']
    server_optim_c_lr = params['server_optim_c_lr']
    server_batch_size = params['server_batch_size']

    c_model_name = params['c_model']
    c_epochs = params['c_epochs']
    c_momentum = params['c_momentum']
    c_lr = params['c_lr']
    client_batch_size = params['client_batch_size']
    clients_per_iter = params['clients_per_iter']

    fltrust_root_iid = params['fltrust_root_iid']
    dba_poisoning_per_batch = params.get('dba_poisoning_per_batch', 0.2)
    
    client_list = list(range(client_num))
    class_list = list(range(fl_class_num))


    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        logger.info('Folder already exists')
    
    t0 = time.time()
    
    logger.setLevel(logging.INFO)
    logger_filehandler = FileHandler(f'{exp_dir}/L_{timestamp}.log')
    logger_filehandler.setLevel(logging.INFO)
    logger_filehandler.setFormatter(Formatter('%(asctime)s %(message)s (%(filename)s)', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(logger_filehandler)
    
    
    # create the FL data for client(s)
    # control n and label_skew_alpha or p-degree can get Desired non-iid and fully use dataset
    fl_data_dict, labels_per_client_dict, label_count_per_client_dict = \
        create_data_fl_clients(client_num, noniid_type, label_skew_alpha, p_degree, dirichlet_alpha, fl_train, fl_class_num)

    # clients selection per iteration
    clients_selection_csv_header = ["iter"] + client_list
    clients_selection_csv = CsvLogging(f"{timestamp}_clients_selections", exp_dir, clients_selection_csv_header)

    # selection mistakes
    mistakes_csv_header = ["iter", "mistakes", "selections"]
    mistakes_csv = CsvLogging(f"{timestamp}_mistakes", exp_dir, mistakes_csv_header)


    if defense == 'AGGR' and aggr == 'fltrust':
        # alwasy IID rootdataset for FLTrust
        root_ds = create_fl_trust_root_dataset(fl_train,
                                               fltrust_root_iid,
                                               noniid_type='label_skew', 
                                               non_iid_param=1.0,
                                               fl_class_num=fl_class_num, 
                                               sample_size=100)

    
    # malicious attack data modification
    if if_attack:
        fl_data_dict, malicious_client_ids, mali_labels = create_malicious_data(client_num, 
                                                                         malicious_num, 
                                                                         fl_data_dict, 
                                                                         attack_method, 
                                                                         image_size,
                                                                         fl_class_num)
        
        logger.info(f"Malicious client(s): {malicious_client_ids} attack with {attack_method}, on class label {mali_labels}")
        
        # count class labels for malicious and bengin clients
        mali_label_list=[]
        beni_label_list=[]
        for client_id, label_list in labels_per_client_dict.items():
            if client_id in malicious_client_ids:
                mali_label_list.append(label_list)
            else:
                beni_label_list.append(label_list)
        mali_label_list = np.concatenate(mali_label_list, axis=None).tolist()
        beni_label_list = np.concatenate(beni_label_list, axis=None).tolist()
        mali_label_count = Counter(mali_label_list)
        beni_label_count = Counter(beni_label_list)
        logger.info(f"Malicous client(s) label counts: {sorted(mali_label_count.items())}")
        logger.info(f"Benign client(s) label counts: {sorted(beni_label_count.items())}")
        
        class_dominance = True

        try:
            for class_id in list(mali_label_count.keys()):
                if beni_label_count[class_id] <= mali_label_count[class_id]:
                    class_dominance = False
                    break
        
            logger.info(f"Benign dominance malicous: {class_dominance}")
        except:
            logger.info(f"Benign dominance malicous error")
    else:
        malicious_client_ids = [] # malicious clients' ids
        mali_labels = []

    if attack_method == 'FedOda':
        # create the pooled dataset for FedOda
        pooled_ds = []
        for client_id, client_data in fl_data_dict.items():
            if client_id in malicious_client_ids:
                pooled_ds+=client_data
        
        label_distribution = count_labels(pooled_ds)
        logger.info(f"pooled dataset label distribution: {label_distribution}")
        
        # pretrain the model to get attack direction
        model_fn = partial(get_model(c_model_name), num_classes=fl_class_num, dataset=params['fl_task'])
        FedOda_ = FedOda(model_fn=model_fn, 
                         device=device, 
                         learning_rate=0.01, 
                         scaling_factor=8,
                         reduce_factor=0.8,
                         increase_factor=1.05)
        
        max_attack_l2_dist, max_attack_acc, attack_accuracy_list, l2_distances, modelt_acc = \
            FedOda_.pretrain(iters=20, pooled_data=pooled_ds, test_data=fl_test)
        
        logger.info(f"max_attack_l2_dist: {max_attack_l2_dist:.3f}, max_attack_acc: {max_attack_acc:.3f}")
        logger.info(f"attack_accuracy_list: {[round(x, 3) for x in attack_accuracy_list]}")
        logger.info(f"l2_distances: {[round(x, 3) for x in l2_distances]}")
        logger.info(f"modelt_acc: {modelt_acc}")
    else:
        FedOda_ = None
        
    loss_acc_csv_header = ["iter", "loss", "accuracy"]
    loss_acc_csv = CsvLogging(f"{timestamp}_loss_acc", exp_dir, loss_acc_csv_header)

    attack_success_rate_csv_header = ['iter'] + mali_labels
    attack_success_rate_csv = CsvLogging(f"{timestamp}_attacks", exp_dir, attack_success_rate_csv_header)

    c_weights_csv_header = ["iter"] + client_list
    c_weights_csv = CsvLogging(f"{timestamp}_c_weights", exp_dir, c_weights_csv_header)

    # record the training
    clients_eval_hist = []
    loss_vail_hist = []
    acc_vail_hist = []
    per_class_acc_vail_hist = []
    selected_clients_per_class_hist = []
    attack_success_hist = []
    mistake_hist = []

    # create plot dir
    plot_dir=f"{exp_dir}/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # create image dir
    img_dir=f"{exp_dir}/g_model_images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    

    # create models 
    mco_dict = create_fl_models(client_num, device, c_model_name,  c_lr, server_optim_c_lr, c_momentum,
                                num_classes=fl_class_num,
                                dataset=params['fl_task'])

    # send main to all clients
    mco_dict = send_main_to_clients(client_num, mco_dict)
    benign_client_ids = []
        
    #fl train
    for iter in range(fl_iter_num):
        # fl training
        client_selection = random.sample(client_list, clients_per_iter)
        client_selection.sort()
        logger.info(f"FL iteration: {iter}, selected clients: {client_selection}")
        
        # client_selection csv logging
        client_selection_dict, _ = create_client_weights(client_selection, client_selection, client_list)
        clients_selection_csv.append_save_csv([iter] + list(dict(sorted(client_selection_dict.items())).values()))
        
        logger.info("local training for clients' C models (AGR)")
        
        # local training for clients' C models including attack
        c_grad_dict = fl_iteration(iter, client_selection, malicious_client_ids,
                                   attack_method, c_epochs, fl_data_dict, mco_dict, 
                                   client_batch_size, dba_poisoning_per_batch, 
                                   FedOda_class = FedOda_, device=device)


        server_iter = 0
        fid_avg = 0
        t1 = time.time()
        
        server_iter += 1
        if defense == 'AGGR':
            if aggr == 'fedavg':
                c_grad = fedavg(c_grad_dict)
            elif aggr == 'geometric_median':
                c_grad, client_weights = geo_med_aggr(c_grad_dict, device)
                c_weights_csv.append_save_csv([iter] + list(dict(sorted(client_weights.items())).values()))
            elif aggr == 'coordinate_wise_median':
                c_grad = coordinate_wise_median(c_grad_dict)
            elif aggr == 'trimmed_mean':
                # best case server knows malicious client num
                m = client_num - malicious_num
                c_grad = trimmed_mean(c_grad_dict, m)
            elif aggr == 'krum':
                # best case half the cliets are malicious
                # orgainal Krum select one client
                f = client_num//2 - 1
                m = 1 # single krum
                c_grad, krum_clients = krum_aggr(c_grad_dict, f, m)
                benign_client_ids = krum_clients
                print(f"krum clients: {krum_clients}")
            elif aggr == "multi-krum":
                f = client_num//2 - 1
                m = client_num//2 - 1 # multi krum
                c_grad, krum_clients = krum_aggr(c_grad_dict, f, m)
                benign_client_ids = krum_clients
                print(f"multi-krum clients: {krum_clients}")                       
            elif aggr == 'flame':
                # flame works with model weight, convert grad to model weight
                server_c_sd = mco_dict[-1]["c_model"].state_dict()
                c_grad, flame_clients = flame_aggr(server_c_sd, c_grad_dict, exp_dir=exp_dir, iter=iter)
                benign_client_ids = flame_clients
                print(f"flame clients: {flame_clients}")
            elif aggr == 'normbound':
                print("aggr: normbound")
                server_c_sd = mco_dict[-1]["c_model"].state_dict()
                c_grad = normbound_aggr(server_c_sd, c_grad_dict)
            elif aggr == 'fltrust':
                # train server with root dataset
                server_c_grad = fltrust_server_iteration(iter, c_epochs,
                                                            root_ds, mco_dict, batch_size=5)
                
                # caculate trust score
                FLTrustTotalScore = 1e-9
                trust_score_list = []
                clip_value_list = []
                
                for client_id in client_selection:
                    client_trust_score, client_clipped_value = cosScoreAndClipValue(server_c_grad, c_grad_dict[client_id])
                    FLTrustTotalScore += client_trust_score
                    trust_score_list.append(client_trust_score)
                    clip_value_list.append(client_clipped_value)
                
                trust_score_list = [x / FLTrustTotalScore  for x in trust_score_list]
                # weighted average of grad
                fltrust_weights_list = [a*b for a,b in zip(trust_score_list, clip_value_list)]
                fltrust_weights = dict(zip(client_selection, fltrust_weights_list))
                c_weights_csv.append_save_csv([iter] + list(dict(sorted(fltrust_weights.items())).values()))
                c_grad = grad_weighted_sum(c_grad_dict, fltrust_weights)
            
            
            # Update the models in mco_dict directly
            grad_add_to_model(mco_dict[-1]["c_model"], aggr_c_lr, c_grad) 
            
        
        # for targeted attack test on one of the malicious client
        # validation on server's C model
        if attack_method in ['clear_label_backdoor', 'backdoor', '2_label_flipping', 'dba']:
            _, _, _,  client_asr = fl_validation(mco_dict[malicious_client_ids[0]]["c_model"], 
                                                    mco_dict[malicious_client_ids[0]]["c_criterion"], 
                                                    fl_test, 
                                                    server_batch_size,
                                                    fl_class_num,
                                                    attack_method, 
                                                    mali_labels)
            logger.info(f"iter {iter}: {malicious_client_ids[0]} client_asr: {client_asr.values()}")

        # validation on server's C model
        test_loss, test_accuracy, per_class_accuracy,  attack_success_rate = fl_validation(mco_dict[-1]["c_model"], 
                                                                            mco_dict[-1]["c_criterion"], 
                                                                            fl_test, 
                                                                            server_batch_size,
                                                                            fl_class_num,
                                                                            attack_method, 
                                                                            mali_labels,
                                                                            device=device)
        loss_vail_hist.append(test_loss)
        acc_vail_hist.append(test_accuracy)
        per_class_acc_vail_hist.append(per_class_accuracy)
        attack_success_hist.append(attack_success_rate)
        loss_acc_csv.append_save_csv([iter, test_loss, test_accuracy])
        attack_success_rate_csv.append_save_csv([iter] + list(attack_success_rate.values()))

        logger.info(f"iter {iter}: C-Model, vali loss: {test_loss:.6f}, accuracy: {test_accuracy:.6f}")
        if len(attack_success_rate) != 0:
            logger.info(f"attack_success:{attack_success_rate}")
        
        # for FLAME and FedEval, check the clients selection's malicious rate
        if benign_client_ids:
            mistake_rate =  len(set(benign_client_ids) & set(malicious_client_ids)) / len(benign_client_ids)
            mistakes_csv.append_save_csv([iter, mistake_rate])

        
        e = int((time.time()-t1)/(iter+1) *
                (fl_iter_num - iter))
        print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                "[{:.2f}%]\n".format(iter / fl_iter_num * 100))
        print("Elapsed Time this round:", "{:.2f} sec".format(time.time() - t1))
        logger.info(f"exp total running time: {datetime.timedelta(seconds=(time.time() - t0))}")
        
        # send main to all clients
        mco_dict = send_main_to_clients(client_num, mco_dict)


    # save server models
    path_c_model=f"{exp_dir}/c_model_{iter}.pt"
    torch.save(mco_dict[-1]["c_model"].state_dict(), path_c_model)

    logger.info(f"-----------{timestamp} FL Experiment Finished-----------")
    logger.handlers.clear()
    
    return loss_vail_hist, acc_vail_hist, per_class_acc_vail_hist, \
        selected_clients_per_class_hist, clients_eval_hist, attack_success_hist, fid_avg

