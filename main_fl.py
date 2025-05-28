import os
import yaml
import argparse
from fl_train import fl_training
from data_process import data_process
import logging
from logging import FileHandler
from logging import Formatter
from utilities import timestamp
import shutil
from csv_logging import CsvLogging
import sys
from logging import StreamHandler

m_logger = logging.getLogger("master_logging")


def main():
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    current_time = timestamp(filename=True)
    
    with open(f'{args.params}', 'r') as f:
        params = yaml.safe_load(f)
    
    fl_task = params['fl_task']
    defense = params['defense']
    aggr = params['aggr']
    image_size = params['image_size']
    if_series = params['if_series']
    clients_eval_mode = params['clients_eval_mode']
    if if_series:
        exp_dir = f"logs_fl/{fl_task}/S{fl_task}_{current_time}_{defense}_{aggr}_{clients_eval_mode}"
    else:
        exp_dir = f"logs_fl/{fl_task}/{fl_task}_{current_time}_{defense}_{aggr}_{clients_eval_mode}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    shutil.copy(args.params, f"{exp_dir}/{fl_task}_FL_{current_time}.yaml")
        
    # Clear existing handlers
    if m_logger.hasHandlers():
        m_logger.handlers.clear()
    
    # Setup logger
    m_logger.setLevel(logging.INFO)
    
    # File handler
    m_logger_filehandler = FileHandler(f'{exp_dir}/S_{current_time}.log')
    m_logger_filehandler.setLevel(logging.INFO)
    m_logger_filehandler.setFormatter(Formatter('%(asctime)s %(message)s (%(filename)s)', '%Y-%m-%d %H:%M:%S'))
    m_logger.addHandler(m_logger_filehandler)
    
    # Stream (console) handler
    stream_handler = StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(Formatter('%(asctime)s %(message)s (%(filename)s)', '%Y-%m-%d %H:%M:%S'))
    m_logger.addHandler(stream_handler)
    
    # Optional: prevent propagation to root
    m_logger.propagate = False
    
    m_logger.info(f"Defense: {defense} - {aggr}")

    ds_percentage = params.get('ds_percentage', 1)
    
    # FL process
    train_data, test_data = data_process(params['fl_task'], image_size, percentage=ds_percentage)
    
    m_logger.info(f"Created dataset for FL task: {params['fl_task']}")
    
    if not if_series:
        m_logger.info(f"One exp, time:{current_time}")
        fl_w_visualization(train_data, test_data, params, image_size, exp_dir, fl_task, defense, aggr, current_time)
    else:
        if params['noniid_type'] == 'label_skew':
            non_iid_params = params['label_skew_alphas']
        elif params['noniid_type'] == "p-degree":
            non_iid_params = params['p-degrees']
        elif params['noniid_type'] == 'dirichlet':
            non_iid_params = params['dirichlet_alphas']
        attack_methods = params['attack_methods']
        
        total_exp = len(non_iid_params) * len(attack_methods)
        master_csv_header = ["exp_num","label_skew_alpha", "attack_method", "loss", "acc", "fid", "attack_1", "attack_2"]
        master_csv = CsvLogging(f"{fl_task}_{defense}_{aggr}_master", exp_dir, master_csv_header)
        exp_num = 1
        
        for non_iid in non_iid_params:
            for attack_method in attack_methods:
                if params['noniid_type'] == 'label_skew':
                    params["label_skew_alpha"] = non_iid
                elif params['noniid_type'] == "p-degree":
                    params['p-degree'] = non_iid
                elif params['noniid_type'] == 'dirichlet':
                    params['dirichlet_alpha'] = non_iid

                params['attack_method'] = attack_method
                current_time = timestamp(filename=True)
                m_logger.info(f"{exp_num}/{total_exp}, time:{current_time}, attack_method:{attack_method}")
                sub_exp_dir = f"{exp_dir}/{current_time}_{attack_method}_{non_iid}"
                
                loss_vail_last_iter, acc_vail_last_iter, per_class_acc_vail_last_iter, attack_success_last_iter, fid = \
                    fl_w_visualization(train_data, test_data, params, image_size, sub_exp_dir, fl_task, defense, aggr, current_time)
                
                
                master_csv.append_save_csv([exp_num,
                                            non_iid, attack_method, 
                                            loss_vail_last_iter, acc_vail_last_iter, fid] + \
                                            list(attack_success_last_iter.values()))
                    
                exp_num += 1
        

def fl_w_visualization(train_data, test_data, params, image_size, exp_dir, fl_task, defense, aggr, timestamp):
        loss_vail_hist, acc_vail_hist, per_class_acc_vail_hist, \
        selected_clients_per_class_hist, self_eval_hist, attack_success_hist, fid = \
            fl_training(train_data, test_data, params, image_size, exp_dir, timestamp)
        
        m_logger.info(f"loss_vail\n{loss_vail_hist}")
        m_logger.info(f"ac vail\n{acc_vail_hist}")
        m_logger.info(f"attack_success_hist\n{attack_success_hist}")
           
        return loss_vail_hist[-1], acc_vail_hist[-1], per_class_acc_vail_hist[-1], attack_success_hist[-1], fid

if __name__ == '__main__':
    main()