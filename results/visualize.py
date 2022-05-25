"""
可视化分析
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from seqeval.metrics import accuracy_score, precision_score, classification_report
import math


OURS_RESULT_SAVE_DIR =  "/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/snips/by_count_10/bert_self_train_debug_55"
OURS_BEST_EPOCH = 17   # 17

BASELINE_RESULT_SAVE_DIR = "/mnt/2/ljc/workspace/self-training-for-slot-filling/results/snips/by_count_10/bert_self_train_debug_10"
BASELINE_BEST_EPOCH = 3

FIG_SAVE_PATH = 'slot_num_of_best_epoch.png'

SLOT_TYPE_LIST = ['object_type', 'object_name', 'playlist', 'rating_value', 'timeRange', 'artist', 'music_item', 'city', 'restaurant_type', 'spatial_relation', 'rating_unit', 'playlist_owner', 'best_rating', 'state', 'party_size_number', 'object_select', 'country', 'movie_name', 'service', 'movie_type', 'year', 'location_name', 'entity_name', 'sort', 'object_location_type', 'condition_temperature', 'condition_description', 'restaurant_name', 'party_size_description', 'object_part_of_series_type', 'geographic_poi', 'current_location', 'served_dish', 'track', 'cuisine', 'album', 'facility', 'genre', 'poi']


def cal_variance(slot_num_dic):
    """ 计算方差, 返回slot总量和方差 """
    nums = list(slot_num_dic.values())
    arr = np.array(nums)
    arr_sum = np.sum(arr)
    arr = arr / arr_sum
    arr_var = np.var(arr)
    return arr_var, arr_sum
    

def stat_each_epoch_slot_distribution_variance(seq_out_file_path):
    """ 统计每轮epoch当中训练数据的槽位数量 """

    slot_num_dic = {}
    with open(seq_out_file_path, 'r', encoding='utf8') as f:
        for line in f:
            labels = line.strip().split(' ')
            for label in labels:
                if label.startswith("B-"):
                    slot = label.split("-")[1]
                    slot_num_dic[slot] = slot_num_dic.get(slot, 0) + 1
    return slot_num_dic
    

def plot_slot_distribution(x1, y1, x2, y2):
    """ 绘制每轮epoch中槽位分布变化曲线 """

    fig = plt.figure(num=1, figsize=(12,6), dpi=500)
    plt.style.use('seaborn-darkgrid')
    ax = fig.add_subplot(111)

    # ax.set_xlim(1,4)
    # ax.set_ylim(1000,500000)


    ax.plot(x1, y1,'-o', color='#FF8C00', label='SIAST')
    ax.plot(x2, y2,'-o', color='#4169E1', label='ClassicST')

    ax.set_xticks([1,2,3])
    # ax.set_yticks([20,30,40,50,60,70])
    # ax.set_yticks(np.linspace(40,90,6))
    ax.set_xticklabels(['majority', 'medium', 'minority'], fontsize=20)
    # ax.set_yticklabels(['20','30','40','50','60','70'], fontsize=20)
    # ax.set_yticklabels(['40', '50', '60', '70', '80', '90'], fontsize=20)
    # ax.set_title('Results of Mis Experiment', fontsize=20, 
    # # fontweight='black',
    # pad=10)
    ax.set_xlabel('total',fontsize=20)
    ax.set_ylabel('Variance',fontsize=15)
    # for i in range(len(x)):
    #     ax.text(x[i]-.15, y1[i]+1, y1[i], fontsize=20)
    # for i in range(len(x)):
    #     ax.text(x[i]-.15, y2[i]+1, y2[i], fontsize=20)
    # for i in range(len(x_n)):
    #     ax.text(x_m[i]-.25, y_m[i]+1, y_m[i], fontsize=15)
    # 添加图例
    ax.legend( loc=2, labelspacing=1, handlelength=1.5, fontsize=14, shadow=True)
    plt.show()
    fig.savefig(FIG_SAVE_PATH, bbox_inches='tight', pad_inches=0.05)


def visualize_slot_distribution_valiance():
    """ 可视化槽位分布方差对比图 """
    path1 = os.path.join(OURS_RESULT_SAVE_DIR, "saved_data")

    ours_sum_list = []
    ours_var_list = []
    for i in range(1, OURS_BEST_EPOCH+1):
        seq_out_file_path = os.path.join(path1, f"epoch_{i}", "labeled/seq.out")
        slot_num_dic = stat_each_epoch_slot_distribution_variance(seq_out_file_path)
        var, sum_ = cal_variance(slot_num_dic)
        ours_var_list.append(var)
        ours_sum_list.append(sum_)


    path2 = os.path.join(BASELINE_RESULT_SAVE_DIR, "saved_data")

    base_sum_list = []
    base_var_list = []
    for i in range(1, BASELINE_BEST_EPOCH+1):
        seq_out_file_path = os.path.join(path2, f"epoch_{i}", "labeled/seq.out")
        slot_num_dic = stat_each_epoch_slot_distribution_variance(seq_out_file_path)
        var, sum_ = cal_variance(slot_num_dic)
        base_var_list.append(var)
        base_sum_list.append(sum_)
    
    print(ours_sum_list, ours_var_list)
    print(base_sum_list, base_var_list)

    plot_slot_distribution(ours_sum_list, ours_var_list, base_sum_list, base_var_list)


def visualize_slot_num_distribution():
    """ 绘制数量分布 """
    ours_seq_out_file_path = os.path.join(OURS_RESULT_SAVE_DIR, "saved_data", f"epoch_{OURS_BEST_EPOCH}", "labeled/seq.out")
    ours_slot_num_dic = stat_each_epoch_slot_distribution_variance(ours_seq_out_file_path)

    base_seq_out_file_path = os.path.join(BASELINE_RESULT_SAVE_DIR, "saved_data", f"epoch_{BASELINE_BEST_EPOCH}", "labeled/seq.out")
    base_slot_num_dic = stat_each_epoch_slot_distribution_variance(base_seq_out_file_path)


    
    ours_slot_num_list = []
    base_slot_num_list = []
    for slot in SLOT_TYPE_LIST:
        ours_slot_num_list.append(ours_slot_num_dic[slot])
        base_slot_num_list.append(base_slot_num_dic[slot])

    print(len(ours_slot_num_list))
    print(len(base_slot_num_list))
    
    x = list(range(1, 40))
    print(x)
    
    plot_slot_distribution(x, ours_slot_num_list, x, base_slot_num_list)


def visualize_pseudo_truth_error_ratio():
    """ 绘制labeled data的错误率(pseudo label带来的) """
    ours_pseudo_labeled_train_data_file = os.path.join(OURS_RESULT_SAVE_DIR, "saved_data", f"epoch_{OURS_BEST_EPOCH}", "labeled/seq.out")
    ours_truth_labeled_train_data_file = os.path.join(OURS_RESULT_SAVE_DIR, "saved_data", f"epoch_{OURS_BEST_EPOCH}", "labeled/true.label")

    ours_y_pseudo = []
    with open(ours_pseudo_labeled_train_data_file, 'r', encoding='utf8') as f:
        for line in f:
            ours_y_pseudo.append(line.strip().split(' '))

    ours_y_true = []
    with open(ours_truth_labeled_train_data_file, 'r', encoding='utf8') as f:
        for line in f:
            ours_y_true.append(line.strip().split(' '))
    
    ours_metric_report = classification_report(ours_y_true, ours_y_pseudo, output_dict=True)


    base_pseudo_labeled_train_data_file = os.path.join(BASELINE_RESULT_SAVE_DIR, "saved_data", f"epoch_{BASELINE_BEST_EPOCH}", "labeled/seq.out")
    base_truth_labeled_train_data_file = os.path.join(BASELINE_RESULT_SAVE_DIR, "saved_data", f"epoch_{BASELINE_BEST_EPOCH}", "labeled/true.label")

    base_y_pseudo = []
    with open(base_pseudo_labeled_train_data_file, 'r', encoding='utf8') as f:
        for line in f:
            base_y_pseudo.append(line.strip().split(' '))

    base_y_true = []
    with open(base_truth_labeled_train_data_file, 'r', encoding='utf8') as f:
        for line in f:
            base_y_true.append(line.strip().split(' '))
    
    base_metric_report = classification_report(base_y_true, base_y_pseudo, output_dict=True)
    
  
    head = ['object_type', 'object_name', 'playlist', 'rating_value', 'timeRange', 'artist', 'music_item', 'city', 'restaurant_type', 'spatial_relation', 'rating_unit', 'playlist_owner', 'best_rating']
    medium = ['state', 'party_size_number', 'object_select', 'country', 'movie_name', 'service', 'movie_type', 'year', 'location_name', 'entity_name', 'sort', 'object_location_type', 'condition_temperature']
    tail = ['condition_description', 'restaurant_name', 'party_size_description', 'object_part_of_series_type', 'geographic_poi', 'current_location', 'served_dish', 'track', 'cuisine', 'album', 'facility', 'genre', 'poi']

    ours_head = []
    base_head = []

    ours_medium = []
    base_medium = []

    ours_tail = []
    base_tail = []

    for s in head:
        ours_head.append(1 - ours_metric_report[s]['precision'])
        base_head.append(1 - base_metric_report[s]['precision'])
    for s in medium:
        ours_medium.append(1 - ours_metric_report[s]['precision'])
        base_medium.append(1 - base_metric_report[s]['precision'])
    for s in tail:
        ours_tail.append(1 - ours_metric_report[s]['precision'])
        base_tail.append(1 - base_metric_report[s]['precision'])

    
    x = list(range(1, 4))
    ours = [sum(ours_head)/len(ours_head), sum(ours_medium)/len(ours_medium), sum(ours_tail)/len(ours_tail)]
    base = [sum(base_head)/len(base_head), sum(base_medium)/len(base_medium), sum(base_tail)/len(base_tail)]
    
    plot_slot_distribution(x, ours, x, base)



if __name__ == "__main__":
    # visualize_slot_distribution_valiance()
    # plot_slot_num_distribution()
    visualize_pseudo_truth_error_ratio()


