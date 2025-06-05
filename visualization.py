import numpy as np  
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
import torch
from collections import Counter
import matplotlib
import os
matplotlib.use('Agg')

def same_label_lr_class_bars(iter, sensor_classifier, sensor_name, total_clients, class_num, save_dir, filename):
    # use for sensor and fl task share same labels, but different feature e.g. MNIST & SEMEION
    acc_dict = {}
    for c_id in range(total_clients):
        acc_list = []
        for t_class in range(class_num):
            acc = sensor_classifier[iter][c_id][t_class][sensor_name][0]
            acc_list.append(acc)
        acc_dict[c_id] = acc_list
    class_list = list(range(class_num))
    client_list = list(acc_dict.keys())
    
    x = np.arange(class_num)  # the label locations
    width = 1/(total_clients+1)  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
    
    for attribute, measurement in acc_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label="client"+str(attribute))
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Sensor {sensor_name} (iter {iter})')
    ax.set_xticks(x + width, ["class" + str(i) for i in class_list])
    ax.legend(loc='upper left', ncol=len(client_list))
    ax.set_ylim(0, 1.2)
    
    fig.savefig(f'{save_dir}/sensor_acc_{filename}.png')
    plt.close()

    

def matrix_plot(dist_matrix, title, filename, x_label, y_label, save_dir, client_ids):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(dist_matrix, interpolation='nearest', cmap='RdYlBu')
    fig.colorbar(cax)

    ax.set_xticks(range(len(client_ids)))
    ax.set_yticks(range(len(client_ids)))
    ax.set_xticklabels(client_ids, rotation=90)
    ax.set_yticklabels(client_ids)
    
    plt.xlabel(x_label) 
    plt.ylabel(y_label) 
    plt.title(f"{title}") 
    plt.savefig(f'{save_dir}/{filename}.png')
    plt.close()


def dist_single_center(iter, sensor_classifier, sensor_name, total_clients, class_num, dist, save_dir, filename):
    class_dict = {}
    for t_class in range(class_num):
        class_dict[t_class] = []
        z_list = []
        for i in range(total_clients):
            z = sensor_classifier[iter][i][t_class][sensor_name][2]
            z_list.append(z)
        z_geo = torch.cat(z_list, 0)
        
        for j in range(total_clients):
            zj = sensor_classifier[iter][j][t_class][sensor_name][2]
            class_dict[t_class].append(float(torch.norm(torch.cdist(zj, z_geo, p=2))))
    
    print("class_dict", class_dict)
    
    pass

def kl_bars(tt, ti, filename, save_dir):
    X = list(range(10))
    train_test = tt.values()
    train_invert = ti.values()
    
    X_axis = np.arange(len(X)) 
    
    plt.bar(X_axis - 0.2, train_test, 0.4, label = 'Normal') 
    plt.bar(X_axis + 0.2, train_invert, 0.4, label = 'Invert') 
    
    plt.xticks(X_axis, X) 
    plt.xlabel("Labels") 
    plt.ylabel("KL Divergence") 
    plt.title("KL Divergence, Normal Test vs. Invert Test") 
    plt.legend() 
    plt.savefig(f'{save_dir}/kld_bars_{filename}.png')
    plt.close()


def accuracy_bars(tt, ti, filename, save_dir): 

    X = list(range(10))
    train_test = [i * 100 for i in tt.values()]
    train_invert = [i * 100 for i in ti.values()]
    
    X_axis = np.arange(len(X)) 
    
    plt.bar(X_axis - 0.2, train_test, 0.4, label = 'Normal', color = 'tab:green') 
    plt.bar(X_axis + 0.2, train_invert, 0.4, label = 'Invert', color='tab:red') 
    
    plt.xticks(X_axis, X) 
    plt.xlabel("Labels") 
    plt.ylabel("Accuracy") 
    plt.yticks(np.arange(0, 110, 10))
    plt.title("Classification Accuracy, Normal Test vs. Invert Test") 
    plt.legend(loc='upper right')
    plt.savefig(f'{save_dir}/accuracy_bar_{filename}.png')
    plt.close()
    
def plot_curve(data, name, filename, save_dir):
    plt.xlabel("Iterations") 
    plt.ylabel(name)
    plt.plot(data)
    plt.title(f"{name}")
    plt.savefig(f'{save_dir}/{name}_curve_{filename}.png')
    plt.close()


def plot_multi_curves(data_dict, title, filename, save_dir):
    """
    Plot multiple curves in one figure.

    Args:
        data_dict (dict): Keys are labels, values are lists of data points.
        title (str): Plot title.
        filename (str): File name (no extension).
        save_dir (str): Directory to save the plot.
    """
    plt.figure()
    for label, values in data_dict.items():
        if values is not None and any(v is not None for v in values):
            plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel("Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()


def plot_twin_curves(data_dict, title, filename, save_dir, 
                    ylabel1="Left Y-Axis", ylabel2="Right Y-Axis"):
    plt.figure(figsize=(10, 6))
    
    # Plot first curve
    if len(data_dict) < 1:
        print(f"WARNING: No data to plot for {title}")
        return
    
    # Get first key-value pair
    key1, values1 = next(iter(data_dict.items()))
    color1 = 'tab:blue'
    ax1 = plt.gca()
    line1, = ax1.plot(values1, label=key1, color=color1)
    ax1.set_xlabel('Round')
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot second curve if exists
    if len(data_dict) > 1:
        key2, values2 = list(data_dict.items())[1]
        color2 = 'tab:red'
        ax2 = ax1.twinx()
        line2, = ax2.plot(values2, label=key2, color=color2)
        ax2.set_ylabel(ylabel2, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Only create legend if we have valid labels
        if key1 and key2:
            plt.legend([line1, line2], [key1, key2])
    else:
        if key1:
            plt.legend([line1], [key1])
    
    plt.title(title)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path)
    print(f"DEBUG: Saved plot to {save_path}")  # Confirmation message
    plt.close()
    
    
def per_class_accuracy_plot(per_class_acc_vail_hist, filename, title, save_dir):
    iters = len(per_class_acc_vail_hist)
    class_ids = per_class_acc_vail_hist[0].keys()
    class_accuarcy = {class_id:[] for class_id in class_ids}
    for iter in range(iters):
        for class_id, accu in per_class_acc_vail_hist[iter].items():
            class_accuarcy[class_id].append(accu)
    
    x = list(range(iters))
    for class_id, accu_list in class_accuarcy.items():
        plt.plot(x, accu_list, label = "label" + str(class_id))
    
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, iters, step=1)) 
    # plt.legend(fontsize='medium')
    plt.title(title)
    plt.savefig(f'{save_dir}/{filename}_per_class_accuracy.png')
    plt.close()
  
    
def client_class_counts(selected_clients_per_class_hist, class_num, malicous_clients, malicous_classes):
    summary_dict = {key: [] for key in range(class_num)}
    count_dict = {key: 0 for key in range(class_num)}
    total_counts = 0
    for i in range(len(selected_clients_per_class_hist)):
        for class_id, client_ids in selected_clients_per_class_hist[i].items():
            summary_dict[class_id] += client_ids
            total_counts += len(client_ids)

    for class_id in range(class_num):
        count_ = Counter(summary_dict[class_id])
        count_dict[class_id] = dict(sorted(count_.items()))
    
    mistake_count = 0
    for malicious_class in malicous_classes:
        for malicous_id in malicous_clients:
            if malicous_id in count_dict[malicious_class]:
                mistake_count += count_dict[malicious_class][malicous_id]
    return count_dict, mistake_count, total_counts
        