import os
import yaml
import argparse
from data_process import data_process
from sensor_train import sensor_train
from fl_train import fl_training
from utilities import timestamp
import shutil
import logging
logger = logging.getLogger(__name__)
# read yaml file, based on the yaml file
# 1. create sensor(s)
# 2. simulation for FL with 1 client, test the result of the evluations


def main():
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    series_time = timestamp(filename=True)

    with open(f'{args.params}', 'r') as f:
        params = yaml.safe_load(f)

    sensor_name = params['sensor']
    exp_dir = f"logs_sensor/{sensor_name}_{params['adversarial_train']}_{series_time}"
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    shutil.copy(args.params, f"{exp_dir}/{sensor_name}_{series_time}.yaml")
    
    logging.basicConfig(format='%(asctime)s %(message)s (%(filename)s)', datefmt='%m/%d/%Y %I:%M:%S', 
                        filename=f'{exp_dir}/{series_time}.log', level=logging.INFO)
    logger.addHandler(logging.StreamHandler())
    
    logger.info(f'Sensor training starts on {sensor_name}')
    
    train_data, test_data = data_process(sensor_name, image_size=[3,224,224], sensor_train=True)

    path_vae = sensor_train(train_data, test_data, sensor_name, params, series_time, exp_dir)
        
    logger.info(f"{sensor_name} dataset, VAE:{path_vae}")
    # logger.info(f"{sensor_name} dataset,  LR:{path_lr}")
    # logger.info(f"{sensor_name} dataset, label distribution dict({master_sensor}):{path_label_dist}")
    # logger.info(f"{sensor_name} dataset, label mapping dict:{path_mapping_dict}")

if __name__ == '__main__':
    main()