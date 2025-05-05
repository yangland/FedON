import numpy as np  
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
import torch
from collections import Counter
import matplotlib
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

# def dist_permutations(iter, sensor_classifier, sensor_name, total_clients, class_num, dist, save_dir, filename):
#     dist_dict = {}
#     for t_class in range(class_num):
#         dist_matrix = np.array([[0]*total_clients]*total_clients)
#         for i in range(total_clients):
#             a_z = sensor_classifier[iter][i][t_class][sensor_name][2]
#             for j in range(total_clients):
#                 b_z = sensor_classifier[iter][j][t_class][sensor_name][2]
#                 # dist_matrix[i][j] = float(torch.norm(torch.cdist(a_z, b_z, p=2)).cpu().detach())
#                 # dist_matrix[i][j] = float(F.kl_div(a_z, b_z, reduction='sum').cpu().detach())
#                 if dist == "l2-geo":
#                     gm_1 = geometric_median(a_z)
#                     gm_2 = geometric_median(b_z)
#                     dist_matrix[i][j] = float(torch.norm(gm_1 - gm_2).cpu())
#                 elif dist == "wasserstein":
#                     sinkhorn = SinkhornDistance(eps=0.001, max_iter=500)
#                     wass, P, C = sinkhorn(a_z, b_z)
#                     # print("dist wasserstein", wass)
#                     dist_matrix[i][j] = float(wass)
                
#         dist_dict[t_class] = dist_matrix
        
#         fig = plt.figure(figsize=(8, 5))
#         ax = fig.add_subplot(111)
#         cax = ax.matshow(dist_matrix, interpolation='nearest')
#         fig.colorbar(cax)

#         plt.xlabel("client ID") 
#         plt.ylabel("client ID") 
#         plt.title(f"Distance {dist}, Label {t_class}, {sensor_name} (iter {iter})") 
#         plt.savefig(f'{save_dir}/dist_{filename}_label{t_class}.png')
#         plt.close()
#     return dist_dict


# def matrix_plot(dist_matrix, title, filename, x_label, y_label, save_dir):
#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(dist_matrix, interpolation='nearest')
#     fig.colorbar(cax)

#     plt.xlabel(x_label) 
#     plt.ylabel(y_label) 
#     plt.title(f"{title}") 
#     plt.savefig(f'{save_dir}/{filename}.png')
#     plt.close()    
    

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

# def z_distances_comparison(a, b, c, y_label, filename, save_dir):
#     gm = geometric_median(a)
#     dist_a = []
#     dist_b = []
#     dist_c = []
    
#     for i in a:
#         dist_a.append(euclidean(i,gm))
#     for i in b:
#         dist_b.append(euclidean(i,gm))
#     for i in c:
#         dist_c.append(euclidean(i,gm))
        
#     bins = np.linspace(min(min(dist_a), min(dist_b), min(dist_c)), max(max(dist_a), max(dist_b), max(dist_c)), 100)
#     plt.hist(dist_a, bins, alpha=0.5, label='train')
#     plt.hist(dist_b, bins, alpha=0.5, label='test')
#     plt.hist(dist_c, bins, alpha=0.5, label='invert')
#     plt.legend(loc='upper right')
#     plt.title(f"Euclidean distance - label {y_label}: Geo of Train - train, test, invert") 
#     plt.savefig(f'{save_dir}/z_comparison_{y_label}_{filename}.png')
#     plt.close()

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
        