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
from copy import deepcopy
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
from logging import StreamHandler
from aggr.flame import flame_aggr
from aggr.fltrust import cosScoreAndClipValue
from aggr.fedavg_median import coordinate_wise_median, fedavg, trimmed_mean
from aggr.geometric_median import geo_med_aggr
from aggr.krum import krum_aggr
from aggr.normbound import normbound_aggr
import statistics
import sys
from FedOda import FedOda, flatten_model
from visualization import plot_curve, per_class_accuracy_plot, plot_multi_curves, plot_twin_curves
logger = logging.getLogger("individual_logging")

def fl_training(fl_train, fl_test, params, image_size, exp_dir, timestamp):
    # Set random seeds for reproducibility
    random.seed(params.get('seed', 42))
    np.random.seed(params.get('seed', 42))
    torch.manual_seed(params.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.get('seed', 42))
    
    client_num = params['n']
    malicious_num = params['m']
    malicious_fraction = malicious_num/client_num
    global device
    device  = torch.device(f"cuda:{params['cuda_device']}")
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
    
    # global scheduler for the learning rate
    # Parse scheduler config from YAML
    scheduler_name = params.get('c_scheduler', 'none')

    # Create dummy optimizer for global scheduler
    global_optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=c_lr)
    
    class ConstantLR:
        """A dummy scheduler that maintains a constant learning rate."""
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            
        def step(self):
            pass  # Do nothing (keeps LR constant)
        
        def get_last_lr(self):
            return self.base_lrs  # Returns the current LR(s)
        
    # Create global LR scheduler based on config
    if scheduler_name == 'cosine':
        T_max = params.get('c_scheduler_T_max', fl_iter_num)
        global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=T_max)
    elif scheduler_name == 'multistep':
        milestones = params.get('c_scheduler_milestones', [int(0.5 * fl_iter_num), int(0.75 * fl_iter_num)])
        gamma = params.get('c_scheduler_gamma', 0.1)
        global_scheduler = torch.optim.lr_scheduler.MultiStepLR(global_optimizer, milestones=milestones, gamma=gamma)
    else:
        global_scheduler = ConstantLR(global_optimizer)  # Instead of None

    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        logger.info('Folder already exists')
    
    t0 = time.time()
    
    # 💡 Clear existing handlers (important for multi-experiment runs)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # 📁 File handler
    logger_filehandler = FileHandler(f'{exp_dir}/L_{timestamp}.log')
    logger_filehandler.setLevel(logging.INFO)
    logger_filehandler.setFormatter(Formatter('%(asctime)s %(message)s (%(filename)s)', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(logger_filehandler)

    # 🖥️ Stream handler (important to show logs in terminal)
    stream_handler = StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(Formatter('%(asctime)s %(message)s (%(filename)s)', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(stream_handler)

    # ❌ Prevent duplicate prints to root logger
    logger.propagate = False
    
    # create the FL data for client(s)
    # control n and label_skew_alpha or p-degree can get Desired non-iid and fully use dataset
    fl_data_dict, labels_per_client_dict, label_count_per_client_dict = \
        create_data_fl_clients(client_num, noniid_type, label_skew_alpha, p_degree, dirichlet_alpha, fl_train, fl_class_num)
    # logger.info(f"label_count_per_client_dict: {label_count_per_client_dict}")
    
    # clients selection per iteration
    clients_selection_csv_header = ["iter"] + client_list
    clients_selection_csv = CsvLogging(f"{timestamp}_clients_selections", exp_dir, clients_selection_csv_header)

    # selection mistakes
    mistakes_csv_header = ["iter", "mistakes", "posterior"]
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
        pooled_data = []
        malicious_data_lengths = []
    
        # Create a dictionary containing only malicious clients' data
        fl_mali_data_dict = {client_id: client_data 
                            for client_id, client_data in fl_data_dict.items() 
                            if client_id in malicious_client_ids}
    
        for client_id, client_data in fl_data_dict.items():
            if client_id in malicious_client_ids:
                pooled_data.append(client_data)
                malicious_data_lengths.append(len(client_data))
        
        if malicious_data_lengths:
            mali_client_data_size = int(statistics.median(malicious_data_lengths))
            logger.info(f"Median malicious client_data_size: {mali_client_data_size}")
        else:
            mali_client_data_size = 0  # or handle empty case as needed
        
        
        label_distribution = count_labels(pooled_data)
        logger.info(f"pooled dataset label distribution: {label_distribution}")
        
        # pooled dataset from all malicious clients
        model_fn = partial(get_model(c_model_name), 
                           num_classes=fl_class_num,
                           dataset=params['fl_task'])    
        
        pooled_dataset = ConcatDataset(pooled_data)
        pooled_loader = DataLoader(pooled_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        original_vali_loader = DataLoader(fl_test, client_batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        
        # object for FedOda class
        foda = FedOda(model_fn=model_fn, 
                    device=device,
                    pooled_loader=pooled_loader,
                    vali_loader=original_vali_loader,
                    total_clients_num=client_num,
                    malicious_client_ids=malicious_client_ids,
                    fl_mali_data_dict = fl_mali_data_dict,
                    E=params['fedoda']['E'],
                    scaling_l2=params['fedoda']['scaling_factor'],
                    reduce_factor=params['fedoda']['reduce_factor'],
                    increase_factor=params['fedoda']['increase_factor'],
                    threshold=params['fedoda']['threshold'],
                    client_learning_rate=c_lr,
                    client_data_size=mali_client_data_size,
                    client_batch_size=client_batch_size,
                    logger=logger)
    else:
        foda = None
        
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
    mistake_history = []

    # create plot dir
    plot_dir=f"{exp_dir}/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # create image dir
    img_dir=f"{exp_dir}/g_model_images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # create models 
    mco_dict = create_fl_models(client_num, device, c_model_name,  
                                c_lr, server_optim_c_lr, c_momentum,
                                num_classes=fl_class_num,
                                dataset=params['fl_task'],
                                params=params)

    # send main to all clients
    mco_dict = send_main_to_clients(client_num, mco_dict, device)
    benign_client_ids = []
    fid_avg = 0
    
    #fl train
    for iter in range(fl_iter_num):
        t1 = time.time()

        client_selection = random.sample(client_list, clients_per_iter)
        client_selection.sort()
        logger.info(f"FL iteration: {iter}, AGR: {aggr}, selected clients: {client_selection} ")

        # client_selection csv logging
        client_selection_dict, _ = create_client_weights(client_selection, client_selection, client_list)
        clients_selection_csv.append_save_csv([iter] + list(dict(sorted(client_selection_dict.items())).values()))
        
        logger.info("local training for clients' C models (AGR)")
        
        server_before = flatten_model(mco_dict[-1]["c_model"].state_dict())
        
        # STEP 1: Step the global scheduler (safe now, since clients train after this)
        global_scheduler.step()

        # STEP 2: Get the new global LR to be used for this round
        global_lr = global_scheduler.get_last_lr()[0]
        
        # Assign global_lr to each selected client
        for client_id in client_selection:
            for param_group in mco_dict[client_id]["c_optimizer"].param_groups:
                param_group["lr"] = global_lr

        # local training for clients' C models including attack
        c_grad_dict = fl_iteration(iter, client_selection, malicious_client_ids,
                                   attack_method, c_epochs, fl_data_dict, mco_dict,
                                   client_batch_size, dba_poisoning_per_batch,
                                   fedoda_attacker=foda, malicious_fraction=malicious_fraction,
                                   device=device)

        # Step global scheduler
        global_scheduler.step()

        server_after = flatten_model(mco_dict[-1]["c_model"].state_dict())
        server_delta = (server_after - server_before).norm()
        if server_delta > 1e-6:
            logger.warning(f"[Warning] Server model was modified before aggregation! ||Δ|| = {server_delta:.6f}")

        if defense == 'AGGR':
            if aggr == 'fedavg':
                c_grad = fedavg(c_grad_dict)
            elif aggr == 'geometric_median':
                c_grad, geo_weights = geo_med_aggr(c_grad_dict, device)
                c_weights_csv.append_save_csv(
                    [iter] + [v for k, v in sorted(geo_weights.items())]  # Remove .item() since v is already a float
                )
            elif aggr == 'coordinate_wise_median':
                c_grad = coordinate_wise_median(c_grad_dict)
            elif aggr == 'trimmed_mean':
                m = clients_per_iter//2 - 1 
                c_grad = trimmed_mean(c_grad_dict, m)
            elif aggr == 'krum':
                # best case half the cliets are malicious
                # orgainal Krum select one client
                f = clients_per_iter//2 - 1
                m = 1 # single krum
                c_grad, krum_clients, selected_clients_with_scores = krum_aggr(c_grad_dict, f, m)
                benign_client_ids = krum_clients
                logger.info(f"krum clients: {krum_clients}, scores: {selected_clients_with_scores}")
            elif aggr == "multi-krum":
                f = clients_per_iter//2 - 1
                m = clients_per_iter//2 - 1 # multi krum
                c_grad, krum_clients, selected_clients_with_scores = krum_aggr(c_grad_dict, f, m)
                benign_client_ids = krum_clients
                logger.info(f"multi-krum clients: {krum_clients}, scores: {selected_clients_with_scores}")                     
            elif aggr == 'flame':
                # flame works with model weight, convert grad to model weight
                server_c_sd = deepcopy(mco_dict[-1]["c_model"].state_dict())
                c_grad, flame_clients = flame_aggr(server_c_sd, c_grad_dict, exp_dir=exp_dir, iter=iter)
                benign_client_ids = flame_clients
                logger.info(f"flame clients: {flame_clients}, Rejected clients: {set(client_selection) - set(flame_clients)}")
            elif aggr == 'normbound':
                server_c_sd = deepcopy(mco_dict[-1]["c_model"].state_dict())
                c_grad = normbound_aggr(server_c_sd, c_grad_dict)
            elif aggr == 'fltrust':
                # train server with root dataset
                server_c_grad = fltrust_server_iteration(iter, 
                                                        c_epochs=1,
                                                        root_ds=root_ds, 
                                                        mco_dict=mco_dict, 
                                                        batch_size=5, 
                                                        device=device)
                
                
                if has_non_finite_tensor(server_c_grad, name="server_c_grad"):
                    logger.warning(f"[DEBUG] Iter {iter} - Non-finite values in server_c_grad")

                # caculate trust score
                FLTrustTotalScore = 1e-9
                trust_score_list = []
                clip_value_list = []
                
                for client_id in client_selection:
                    client_grad = c_grad_dict[client_id]
                    if has_non_finite_tensor(client_grad, name=f"client {client_id}"):
                        logger.warning(f"[DEBUG] Iter {iter} - Skipping client {client_id} due to non-finite gradients")
                        continue
                    client_trust_score, client_clipped_value = cosScoreAndClipValue(server_c_grad, client_grad)
                    if not math.isfinite(client_trust_score):
                        logger.warning(f"[WARNING] Iter {iter} - Non-finite trust score for client {client_id}")
                    if not torch.isfinite(client_clipped_value).all():
                        logger.warning(f"[WARNING] Iter {iter} - Non-finite clipped value for client {client_id}")

                    trust_score_list.append(client_trust_score)
                    clip_value_list.append(client_clipped_value)
                    FLTrustTotalScore += client_trust_score

                trust_score_list = [x / FLTrustTotalScore  for x in trust_score_list]
                # weighted average of grad
                fltrust_weights_list = [a*b for a,b in zip(trust_score_list, clip_value_list)]
                fltrust_weights = dict(zip(client_selection, fltrust_weights_list))
                c_weights_csv.append_save_csv(
                        [iter] + [v.item() for k, v in sorted(fltrust_weights.items())]
                    )
                float_weights = {k: v.item() for k, v in fltrust_weights.items()}
                logger.info(f"fltrust_weights: {float_weights}")
                c_grad = grad_weighted_sum(c_grad_dict, fltrust_weights)
                
            for k, v in c_grad.items():
                if not torch.is_floating_point(v):
                    logger.debug(f"[DEBUG] Skipping norm for non-floating tensor c_grad[{k}] (dtype={v.dtype})")
                    continue

                if not torch.isfinite(v).all():
                    logger.error(f"[ERROR] Non-finite value in aggregated c_grad[{k}]")
                else:
                    logger.debug(f"[DEBUG] Aggregated Grad Norm - {k}: {v.norm().item():.6f}")


            
            # check nan
            if has_nan_state_dict(c_grad):
                logger.error(f"[ERROR] Iter {iter} - NaN in c_grad. Skipping update.")

            
            # Update the models in mco_dict directly
            grad_add_to_model(mco_dict[-1]["c_model"], 
                              aggr_c_lr, 
                              c_grad,
                              update_bn_buffers=True, 
                              update_batched_tracked=True)  # false and false for ResNet       
        
            # check nan
            if has_nan_state_dict(mco_dict[-1]["c_model"].state_dict()):
                logger.error(f"[ERROR] Iter {iter} - NaN in server state dict. Skipping update.")
        
        
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
        test_loss, test_accuracy, per_class_accuracy, attack_success_rate = fl_validation(mco_dict[-1]["c_model"], 
                                                                            mco_dict[-1]["c_criterion"], 
                                                                            fl_test, 
                                                                            server_batch_size,
                                                                            fl_class_num,
                                                                            attack_method, 
                                                                            mali_labels,
                                                                            device=device)
        
        # >>> EARLY STOPPING CHECK FOR NaN/inf <<<
        if not torch.isfinite(torch.tensor(test_loss)):
            logger.error(f"[ABORT] NaN or Inf detected in validation loss at iteration {iter}.")
            logger.info(f"[ABORT] Final iteration: {iter}. Stopping FL training early due to model divergence.")

            # Save current model
            torch.save(mco_dict[-1]["c_model"].state_dict(), os.path.join(exp_dir, f"server_model_final_iter_{iter}.pt"))

            # Also save logs or metrics if needed
            loss_acc_csv.append_save_csv([iter, test_loss, test_accuracy])
            attack_success_rate_csv.append_save_csv([iter] + list(attack_success_rate.values()))
            
            # Optionally: Save a message to a text file
            with open(os.path.join(exp_dir, "abort_reason.txt"), "w") as f:
                f.write(f"Training aborted at iteration {iter} due to NaN/Inf loss.\n")
                f.write(f"Loss: {test_loss}, Accuracy: {test_accuracy}\n")

            break  # Stop the FL loop
        
        loss_vail_hist.append(test_loss)
        acc_vail_hist.append(test_accuracy)
        per_class_acc_vail_hist.append(per_class_accuracy)
        attack_success_hist.append(attack_success_rate)
        loss_acc_csv.append_save_csv([iter, test_loss, test_accuracy])
        attack_success_rate_csv.append_save_csv([iter] + list(attack_success_rate.values()))

        logger.info(f"iter {iter}: C-Model, vali loss: {test_loss:.6f}, accuracy: {100*test_accuracy:.4f}%")
        if len(attack_success_rate) != 0:
            logger.info(f"attack_success:{attack_success_rate}")
        
        # for FLAME and Krum, check the clients selection's malicious rate
        if benign_client_ids:
            mistake_rate = len(set(benign_client_ids) & set(malicious_client_ids)) / len(benign_client_ids)
            mistakes_csv.append_save_csv([iter, mistake_rate])
            mistake_history.append(mistake_rate)
            logger.info(f"{aggr}-{attack_method} mistake_rate = {mistake_rate*100:.2f}%")

        e = int((time.time()-t1)/(iter+1) * (fl_iter_num - iter))
        print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                "[{:.2f}%]\n".format(iter / fl_iter_num * 100))
        print("Elapsed Time this round:", "{:.2f} sec".format(time.time() - t1))
        logger.info(f"exp total running time: {datetime.timedelta(seconds=(time.time() - t0))}")
        
        # if iter % 10 == 0:
        #     logger.info(f"every 10 iterations, normalize server model weights")
        #     # normalize server model weights
        #     mco_dict[-1]["c_model"].load_state_dict(
        #                 soft_clip_weights(mco_dict[-1]["c_model"].state_dict(), clip_threshold=5.0)
        #             )
        
        # send main to all clients
        mco_dict = send_main_to_clients(client_num, mco_dict, device=device)
        
        # periodic GPU memory cleanup
        if iter % 10 == 0:  # Cleanup every 10 iterations
            torch.cuda.empty_cache()

    # save server models
    path_c_model=f"{exp_dir}/c_model_{iter}.pt"
    torch.save(mco_dict[-1]["c_model"].state_dict(), path_c_model)

    # Extract FedOda attack metrics (if applicable)
    if foda:
        fedoda_metrics = sanitize_attack_metrics(foda.full_history)
        logger.info(f"FedOda metrics1: {fedoda_metrics}")
    else:
        fedoda_metrics = None

    logger.info(f"fedoda_metrics: {fedoda_metrics}")
    logger.info(f"-----------{timestamp} FL Experiment Finished-----------")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.handlers.clear()
    
    plot_fl_results(params, exp_dir, 
                    loss_vail_hist, 
                    acc_vail_hist, 
                    per_class_acc_vail_hist, 
                    attack_success_hist,
                    mistake_history,
                    fedoda_metrics)

    return loss_vail_hist, acc_vail_hist, per_class_acc_vail_hist, \
        selected_clients_per_class_hist, clients_eval_hist, attack_success_hist, fid_avg


def plot_fl_results(params, exp_dir, loss_vail_hist, acc_vail_hist, per_class_acc_vail_hist, 
                    attack_success_hist, mistake_history, fedoda_metrics=None):
    try:
        # Determine non-IID parameter
        if params['noniid_type'] == 'label_skew':
            non_iid_param = params['label_skew_alpha']
        elif params['noniid_type'] == "p-degree":
            non_iid_param = params['p-degree']
        elif params['noniid_type'] == "dirichlet":
            non_iid_param = params['dirichlet_alpha']
        else:
            logger.info("Unknown noniid_type")
            non_iid_param = "unknown"

        fl_task = params['fl_task']
        defense = params['defense']
        aggr = params['aggr']
        attack_method = params['attack_method']

        # Ensure plots directory exists
        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_twin_curves(
            {
                "Loss": loss_vail_hist,
                "Accuracy": acc_vail_hist
            },
            title="Vail Accuracy vs. Loss",
            filename="acc_loss",
            save_dir=plots_dir,
        )

        if len(mistake_history)>0:
            plot_curve(
                mistake_history,
                f"{fl_task}, mistake history, avg mistake {sum(mistake_history)/len(mistake_history):.2f}({attack_method}-{non_iid_param})",
                "mistake_history",
                save_dir=plots_dir
            )

        per_class_accuracy_plot(
            per_class_acc_vail_hist,
            filename=f"{fl_task}_{defense}_{aggr}_accuracy",
            title=f"{fl_task}, Per Class Validation Accuracy Rate ({attack_method}-{non_iid_param})",
            save_dir=plots_dir
        )

        per_class_accuracy_plot(
            attack_success_hist,
            filename=f"{fl_task}_{defense}_{aggr}_attack",
            title=f"{fl_task}, Per Class Attack Success Rate ({attack_method}-{non_iid_param})",
            save_dir=plots_dir
        )

        # --- Plot full history from FedOda ---
        if fedoda_metrics:
            print("\nDEBUG: FedOda metrics keys:", fedoda_metrics.keys())  # Check available keys
            # print("DEBUG: norm_scaling_l2 values:", fedoda_metrics.get("norm_scaling_l2", "MISSING"))
            # print("DEBUG: mapped_scaling_l2 values:", fedoda_metrics.get("mapped_scaling_l2", "MISSING"))      
            
            def try_get(key): return fedoda_metrics.get(key, [])
            
            # Group 1: s_b and s_hat
            # plot_multi_curves(
            #     {
            #         "$s_b$ (estimated benign clients)": try_get("s_b"),
            #         "$\hat{s}$ (estimated total clients)": try_get("s_hat")
            #     },
            #     title="Estimated Client Selection",
            #     filename="clients_estimation",
            #     save_dir=plots_dir
            # )

            # Group 2: benign_norm and server_update_norm
            plot_multi_curves(
                {
                    "benign_norm": try_get("benign_norm"),
                    "server_update_norm": try_get("server_update_norm")
                },
                title="Norms: Benign vs Server Update",
                filename="norm_comparison",
                save_dir=plots_dir
            )

            # Group 3: Cosine distances
            # plot_multi_curves(
            #     {
            #         "cos_d(Δs, Δb')": try_get("cos_d1"),
            #         "cos_d(Δs, Δs')": try_get("cos_d2"),
            #         "cos_d(Δs, Δm)": try_get("cos_d3"),
            #         "cos_d(Δb', Δm)": try_get("cos_d4"),
            #     },
            #     title="Cosine Distances",
            #     filename="cosine_distances",
            #     save_dir=plots_dir
            # )

            # Group 4: L2 distances
            # plot_multi_curves(
            #     {
            #         "L2(Δs, Δb')": try_get("l2_d1"),
            #         "L2(Δs, Δs')": try_get("l2_d2"),
            #         "L2(Δs, Δm)": try_get("l2_d3"),
            #         "L2(Δb', Δm)": try_get("l2_d4"),
            #     },
            #     title="L2 Distances",
            #     filename="l2_distances",
            #     save_dir=plots_dir
            # )

            # Group 5: mean log-likelihood and posterior
            # plot_twin_curves(
            #     {
            #         "Mean Log-Likelihood": try_get("mean_log_likelihood"),
            #         "Posterior P(H1|D)": try_get("posterior")
            #     },
            #     title="Bayesian Inference Metrics",
            #     filename="bayesian_inference",
            #     save_dir=plots_dir,
            # )


            # plot_twin_curves(
            #     {
            #         "scaling_factor": try_get("scaling_factor"),
            #         "Posterior P(H1|D)": try_get("posterior")
            #     },
            #     title="Scaling_factor vs posterior",
            #     filename="scalling_factor",
            #     save_dir=plots_dir,
            # )

            plot_multi_curves(
                {
                    "Reward 1 (r1)": try_get("r1"),
                    "Reward 2 (r2)": try_get("r2"),
                    "Reward 3 (r3)": try_get("r3"),
                    "Accuracy": acc_vail_hist  # Assuming this is accessible
                },
                title="PPO Rewards and Accuracy",
                filename="ppo_rewards_and_accuracy",
                save_dir=plots_dir,
            )


            # Group 6: Scaling factor, epsilon, benign_norm
            plot_multi_curves(
                {
                    "Epsilon": try_get("epsilon"),
                    "Benign Norm": try_get("benign_norm")
                },
                title="Attack Power Parameters",
                filename="attack_parameters",
                save_dir=plots_dir,
            )


            # Group 7: Normalized scaling factors
            plot_twin_curves(
                {
                    "L2 Scaling (norm)": fedoda_metrics["norm_scaling_l2"],
                    "Cos Scaling (norm)": fedoda_metrics["norm_scaling_cos"]
                },
                title="BO: Normalized Scaling Factors",
                filename="bo_normalized_scaling",
                save_dir=plots_dir,
                ylabel1="L2 Scaling (normalized)",
                ylabel2="Cosine Scaling (normalized)"
            )

            # Group 8: Mapped scaling factors
            plot_twin_curves(
                {
                    "L2 Scaling (mapped)": fedoda_metrics["mapped_scaling_l2"],
                    "Cos Scaling (mapped)": fedoda_metrics["mapped_scaling_cos"]
                },
                title="BO: Mapped Scaling Factors",
                filename="bo_mapped_scaling",
                save_dir=plots_dir,
                ylabel1="L2 Scaling (mapped)",
                ylabel2="Cosine Scaling (mapped)"
            )

    except Exception as e:
        logger.info(f"Plotting error: {e}")


def sanitize_attack_metrics(metrics_dict):
    """
    Convert all tensor values in attack metrics to float for plotting.
    Handles lists of tensors and nested data correctly.
    """
    sanitized = {}
    for key, values in metrics_dict.items():
        new_values = []
        for v in values:
            if isinstance(v, torch.Tensor):
                new_values.append(v.detach().cpu().item())
            elif isinstance(v, (float, int)) or v is None:
                new_values.append(v)
            else:
                try:
                    new_values.append(float(v))
                except Exception:
                    new_values.append(None)  # fallback
        sanitized[key] = new_values
    return sanitized


def has_non_finite_tensor(tensor_dict, name=''):
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            logger.warning(f"[WARNING] Non-finite value detected in {name} param '{k}': min={v.min().item()}, max={v.max().item()}")
            return True
    return False


def has_nan_state_dict(state_dict):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                print(f"[DEBUG] NaN in model state key: {k}")
                return True
            if torch.isinf(v).any():
                print(f"[DEBUG] Inf in model state key: {k}")
                return True
    return False

