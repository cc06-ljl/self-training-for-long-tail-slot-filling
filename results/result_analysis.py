import sys
import re
from collections import defaultdict, namedtuple, OrderedDict
from typing import Dict
import json
# from allennlp.nn.util import flattened_index_select
import numpy as np
import math

# ---------------------- 可配置参数 -------------------------------
"""
针对哪次跑出来的模型文件 当中的内容做可视化，来进行参数的配置。最终可视化的图片保存在results/figures/{dataset_name}下。
比如要可视化某次模型跑出来的结果 results/snips/by_count_100/self_train_debug_2/saved_data/epoch_3/test，参数配置如下。注意和文件路径中的某些参量一一对应的关系。
dataset = "snips"
sample_type = "by_count"
quantity = 100
debug = 2
epoch = 3
model_use_bert = False   # 如果是bert跑出来的结果，结果文件路径中是 bert_self_train_debug_2
"""
# 数据集名称
dataset = "snips"  # Few-NERD-supervised
# 第几个debug
debug = 55
# 第几轮epoch
epoch = 17
# 针对哪个集合做可视化,取值：{"test", "current_train_set"}
data_type = "test"
# 哪种采样方式，取值：{"by_count", "by_ratio"}
sample_type = "by_count"
# 针对每个slot选多少条/比率样本
quantity = 10
# 是否使用bert 做embedding
model_use_bert = True
# ---------------------- Over -------------------------------

slots_stats_file = f"/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/data/{dataset}/slot_statistics.json"
if model_use_bert:
    fig_save_dir = f'/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/figures/{dataset}/bert_debug_{debug}'
    macro_f1_save_dir = f'/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/macro_f1/{dataset}/bert_debug_{debug}'
else:
    fig_save_dir = f'/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/figures/{dataset}/debug_{debug}'
    macro_f1_save_dir = f'/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/macro_f1/{dataset}/debug_{debug}'

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

def default_int():
    return defaultdict(int)

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

        # counts the transition rate for every slot type
        self.transitions = defaultdict(default_int)
        self.correct_chunks = []
        self.predicted_chunks = []

def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(lines, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    in_span = False           # both predicted and gold token is in span

    for line in lines:
        line = line.rstrip('\r\n')

        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        # guessed, guessed_type = parse_tag(features.pop())
        # correct, correct_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        guessed, guessed_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        #------- statistics the transition rate
        if in_span:
            if end_correct and end_guessed:
                in_span = False
                counts.transitions[last_correct_type][last_guessed_type] += 1

                counts.correct_chunks.append(last_correct_type)
                counts.predicted_chunks.append(last_guessed_type)

            elif end_correct != end_guessed:
                in_span = False

        #---------------------------------------------

        if in_correct:
            if (end_correct and end_guessed and
                last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
                
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        #------- statistics the transition rate
        if start_correct and start_guessed:
            in_span = True
        #---------------------------------------------

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type
        # if last_guessed_type and last_correct_type:
        #     print(last_guessed_type)
        #     print(last_correct_type)
        #     exit()

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


# def report_transition(transition):
#     all_keys = list(transition.keys())

#     for key, value in transition.item():
#         total = sum(value.values())
from sklearn.metrics import confusion_matrix
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os


def report_confusion_matrix(counts:EvalCounts):
    # all_labels = list(set(counts.predicted_chunks)|set(counts.correct_chunks))
    # labels2idx = {x:i for i, x in enumerate(all_labels)}
    # print(labels2idx)
    # # labels2idx = {i:x for i, x in enumerate(all_labels)}
    # for x in counts.predicted_chunks:
    #     if x == '':
    #         print('-------------------------')
    predicts = np.array([labels2idx[x] for x in counts.predicted_chunks])
    golds = np.array([labels2idx[x] for x in counts.correct_chunks])
    
    np.set_printoptions(threshold=np.inf)
    cm = confusion_matrix(golds, predicts)
    index_list = [2,5,33,35]
    sub_cm = []
    for i in index_list:
        result = []
        for j in index_list:
            result.append(cm[i][j])
        sub_cm.append(result)
    cm = np.array(sub_cm)
    # print(cm)

    # plot_confusion_matrix(cm, list(labels2idx.keys()), f'./figures/{debug}_epoch_{epoch}_{data_type}.png', True)
    # plot_confusion_matrix(cm, list(labels2idx.keys()), f'{fig_save_dir}/quantity_{quantity}_epoch_{epoch}_{data_type}_cm.png', True)
    slot_list = ['playlist', 'artist', 'track', 'album']
    # plot_confusion_matrix(cm, list(labels2idx.values()), f'{fig_save_dir}/quantity_{quantity}_epoch_{epoch}_{data_type}_cm.png', True)
    plot_confusion_matrix(cm, slot_list, f'{fig_save_dir}/quantity_{quantity}_epoch_{epoch}_{data_type}_cm.png', True)
    # with open('matrix_out.csv', 'w')as fout:
    #     for line in cm:
    #         line = [str(x) for x in line]
    #         fout.write(','.join(line)+'\n')


def plot_confusion_matrix(cm,classes, savename, normalize=False, title=f'debug_{debug}_epoch_{epoch}_{data_type}_Confusion Matrix', figsize=(18,15), cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize, dpi=500)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.rcParams['font.size'] = 16
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)# align="center")
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else 'd'
    thresh = cm.max() / 2.
    # for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    #     if cm[i, j] != 0.0:
    #         plt.text(j, i, format(cm[i,j], fmt), horizontalalignment='center',color="white" if cm[i,j]>thresh else "black", fontsize=20)
    # plt.ylabel("true")
    # plt.xlabel("pred")
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.1)


def plot_metrics(by_type, savename):
    """ 绘制每类slot的precision, recall, F1柱状图 """

    slots_tuple_list = sorted(labels2idx.items(), key=lambda item:item[1])   # 索引越小，越靠前(多数类)
    slots_id_descending = np.array([slot[1] for slot in slots_tuple_list])
    slots_name_descending = [slot[0] for slot in slots_tuple_list]
    prec_list = [by_type[slot].prec for slot in slots_name_descending]
    rec_list = [by_type[slot].rec for slot in slots_name_descending]
    f1_score = [by_type[slot].fscore for slot in slots_name_descending]

    fig = plt.figure(figsize=(12,10))
    
    ax1 = fig.add_subplot(111)
    ax1.plot(slots_id_descending, f1_score, 'or-', label="F1 score")
    ax1.plot(slots_id_descending, prec_list, '-2', label="Precision")
    ax1.plot(slots_id_descending, rec_list, '-^', label="Recall")
    plt.title( f'quantity_{quantity}_epoch_{epoch}_{data_type}_metrics')
    ax1.legend(loc=3)
    
    # 将数值显示在图形上
    # for i, (x, y) in enumerate(zip(slots_descending, f1_score)):
    #     plt.text(x, y, f1_score[i], color="black", fontsize=10)  
   
    plt.xticks(slots_id_descending)
    plt.xlabel("class index")
    plt.savefig(savename)

def divide_parts_and_plot_metrics(by_type, savename, divide_type="mean", n=3):
    """ 将39个槽位划分为n个子集，以每个子集中的slots整体分析,绘制P, R折线
    
    :param divide_type 分组模式，"mean" - 平均分组; "scale" - 按0.2,0.3,0.5比例分组
    :param n 分为几组，仅对mean分组模式有效.默认为3.
    
    """
    average_slot_num = len(by_type) // n if (len(by_type) % n == 0) else len(by_type) // n
    # 每个集合包含的slot类数
    # slot_num_per_part = [average_slot_num] * (n-1) + [len(by_type) - average_slot_num * (n-1)]
    train_slots_info = get_train_slots_info(slots_stats_file)
    subsets = []
    # 按类别数平均分为n组
    if divide_type == "mean":
        for i in range(n):
            if i == n - 1:
                subsets.append(train_slots_info[average_slot_num*i:])
            else:
                subsets.append(train_slots_info[average_slot_num*i:average_slot_num*(i+1)])
    
    # 按比例分为3组
    elif divide_type == "scale":
        group_1_num = math.floor(len(train_slots_info) * 0.2)  # 分配到第一组的元素个数（前20%，头部类）
        group_3_num = math.ceil(len(train_slots_info) * 0.5)   # 分配到第三组字列表的元素个数（后50%， 尾部类）
        group_2_num = len(train_slots_info) - group_1_num - group_3_num    # 分配到第二个组列表的元素个数（30%）

        subsets.append(train_slots_info[:group_1_num])
        subsets.append(train_slots_info[group_1_num : group_1_num + group_2_num])
        subsets.append(train_slots_info[group_1_num + group_2_num : ])

    else:
        raise Exception
    # slot_name_subsets = []
    # for slot_info in subsets:
    #     slot_name_subsets.append(list(slot[0] for slot in slot_info))
    
    # 计算每个slot集合的precision, recall
    prec_list = []
    rec_list = []
    f_list = []
    # 计算每个类的 MI-F1
    # for subset in subsets:
    #     fp = 0
    #     fn = 0
    #     tp = 0
    #     for slot_info in subset:
    #         slot_name = slot_info[0]
    #         fp += by_type[slot_name].fp
    #         fn += by_type[slot_name].fn
    #         tp += by_type[slot_name].tp
    #     prec = 0 if tp + fp == 0 else 1.* tp / (tp + fp)
    #     rec = 0 if tp + fn == 0 else 1.* tp / (tp + fn)
    #     f = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    
    # 计算每个类的 MA-F1
    for subset in subsets:
        precin_list = []
        recin_list = []
        fscorein_list = []
        for slot_info in subset:
            slot_name = slot_info[0]
            prec = by_type[slot_name].prec
            rec = by_type[slot_name].rec
            f1_score = by_type[slot_name].fscore
            precin_list.append(prec)
            recin_list.append(rec)
            fscorein_list.append(f1_score)

        prec = np.mean(precin_list)
        rec = np.mean(recin_list)
        f = np.mean(fscorein_list)

        prec_list.append(prec)
        rec_list.append(rec)
        f_list.append(f)

    # 画图
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(111)
    x = list(range(n))
    ax1.plot(x, prec_list, '-2', label="Precision")
    # for a,b in zip(x,prec_list):
    #     plt.text(a,b+0.001,'%.4f'%b,ha = 'center',va = 'bottom',fontsize=12)
    ax1.plot(x, rec_list, '-^', label="Recall")
    # for a,b in zip(x,rec_list):
    #     plt.text(a,b+0.001,'%.4f'%b,ha = 'center',va = 'bottom',fontsize=12)
    ax1.plot(x, f_list, 'or-', label="F1_Score")
    for a,b in zip(x,f_list):
        plt.text(a,b+0.001,'%.4f'%b,ha = 'center',va = 'bottom',fontsize=12)
    plt.title(f'{dataset}_debug_{debug}_quantity_{quantity}_epoch_{epoch}_{data_type}_group_metrics')
    ax1.legend(loc=3)
    plt.xticks(list(range(n)))
    plt.xlabel("subset class index")
    plt.savefig(savename)

    # print()

def report(counts, report_matrix, out=None):
    if out is None:
        out = sys.stdout
    print(counts.transitions)
    # print(counts.t_found_correct)
    # print(counts.predicted_chunks)
    # print(counts.correct_chunks)
    if report_matrix:
        report_confusion_matrix(counts)
    all_correct_num = 0
    all_guessed_num = 0
    if len(counts.t_found_correct)>2:
        for key, value in counts.t_found_correct.items():
            # print('-----------------------')
            # print(key,value, counts.t_found_guessed[key])
            all_correct_num += value
        # print('---------------------------------------')
        # print('---------------------------------------')
        # print('---------------------------------------')
        for key, value in counts.t_found_guessed.items():
            # print('-----------------------')
            # print(key,value)
            all_guessed_num += value
        print(all_correct_num)
        print(all_guessed_num)

    overall, by_type = metrics(counts)
    if report_matrix:
        # save each type F1
        type_F1 = dict()
        for k, v in by_type.items():
            type_F1[k] = v.fscore

        type_F1["macro F1"] = sum(list(type_F1.values()))/len(type_F1)
        with open(f"{macro_f1_save_dir}/quantity_{quantity}_epoch_{epoch}_macro_f1.json", "w", encoding="utf8") as f:
            json.dump(type_F1, f, indent=4)

        plot_metrics(by_type, f'{fig_save_dir}/quantity_{quantity}_epoch_{epoch}_{data_type}_metrics.png')
        divide_type = "mean"
        divide_parts_and_plot_metrics(by_type, f'{fig_save_dir}/quantity_{quantity}_epoch_{epoch}_{data_type}_{divide_type}_group_metrics.png', divide_type=divide_type)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))
    # print(c.t_correct_chunk)
    # print(c.t_found_correct)
    # print(c.t_found_guessed)
    # print(by_type)
    results = {}
    if c.token_counter > 0:
        results["fb1"] = 100.*overall.fscore

    return results

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)

def conll2002_measure(lines, verbose=False, report_matrix=True):
    counts = evaluate(lines, None)
    return report(counts, report_matrix)


def getResult(data_type):
    # if data_type == "test":
    #     resultFile = f"./self_train_debug_BIO_{debug}/saved_data/epoch_{epoch}/test/seq.out"
    #     trueLabelFile = f"./self_train_debug_BIO_{debug}/saved_data/epoch_{epoch}/test/true.label"
    # elif data_type == "current_predict":
    #     resultFile = f"./self_train_debug_BIO_{debug}/saved_data/epoch_{epoch}/labeled/curren_seq.out"
    #     trueLabelFile = f"./self_train_debug_BIO_{debug}/saved_data/epoch_{epoch}/labeled/current_true.label"
    # elif data_type == "current_train_set":
    #     resultFile = f"./self_train_debug_BIO_{debug}/saved_data/epoch_{epoch}/labeled/seq.out"
    #     trueLabelFile = f"./self_train_debug_BIO_{debug}/saved_data/epoch_{epoch}/labeled/true.label"

    # Xcc modified
    if model_use_bert:
        dirs = f"/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/{dataset}/{sample_type}_{quantity}/bert_self_train_debug_{debug}/saved_data/epoch_{epoch}"
    else:
        dirs = f"/mnt/3/prisNlpWorkSpace/self-training-for-slot-filling/results/{dataset}/{sample_type}_{quantity}/self_train_debug_{debug}/saved_data/epoch_{epoch}"

    if data_type == "test":
            resultFile = f"{dirs}/test/seq.out"
            trueLabelFile = f"{dirs}/test/true.label"
    elif data_type == "current_predict":
        resultFile = f"{dirs}/labeled/curren_seq.out"
        trueLabelFile = f"{dirs}/labeled/current_true.label"
    elif data_type == "current_train_set":
        resultFile = f"{dirs}/labeled/seq.out"
        trueLabelFile = f"{dirs}/labeled/true.label"

    with open(resultFile, 'r', encoding="utf8")as fin:
        allRes = fin.readlines()
        allRes = [line.strip().split() for line in allRes]

    with open(trueLabelFile, 'r', encoding="utf8")as fin:
        allLables = fin.readlines()
        allLables= [line.strip().split() for line in allLables]

    bin_lines, type_lines = [], []

    for idx in range(len(allRes)):
        assert len(allRes[idx]) == len(allLables[idx])
        for res, label in zip(allRes[idx], allLables[idx]):
            bin_slot_pred = res[0]
            bin_slot_gold = label[0]
            type_slot_pred = res
            type_slot_gold = label
            bin_lines.append("w" + " " + bin_slot_pred + " " + bin_slot_gold)
            type_lines.append("w" + " " + type_slot_pred + " " + type_slot_gold)

    bin_result = conll2002_measure(bin_lines, report_matrix=False)
    print(bin_result)
    bin_f1 = bin_result["fb1"]
    print(bin_f1)
    print('----------------------------------------------------')
    final_result = conll2002_measure(type_lines)
    print(final_result)
    final_f1 = final_result["fb1"]
    print(final_f1)


def get_train_slots_info(file_path):
    """ 获取按照数量降序排列的(slot, num)列表(文件中有O, 本函数中会去除,返回值将不包括O槽), 返回值形式:[('object_type', 3023), ('object_name', 2789), ...] """
    with open(file_path, 'r', encoding="utf8") as f:
        json_item = json.load(f)
        train_slots_info = sorted(json_item["train"].items(), key=lambda item:item[1], reverse=True)
    # 去除"O"
    train_slots_info.pop(0)
    return train_slots_info


def gen_label2idx_by_descending_order(file_path):
    """ 读取x_slot_statistics.json, 对槽位出现数目降序生成label2idx: index越小, slot数量越多."""
    train_slots_info = get_train_slots_info(file_path)
    train_slots_descending = [slot[0] for slot in train_slots_info]
    labels2idx = {slot:i for i, slot in enumerate(train_slots_descending) if slot != "O"}
    return labels2idx


if __name__ == "__main__":
    # data_type = "current_predict"
    # data_type = "current_train_set"
    # labels2idx = {'object_location_type': 0, 'object_select': 1, 'entity_name': 2, 'album': 3, 'rating_unit': 4, 'restaurant_name': 5, 'movie_name': 6, 'poi': 7, 'sort': 8, 'current_location': 9, 'restaurant_type': 10, 'object_type': 11, 'track': 12, 'playlist_owner': 13, 'city': 14, 'spatial_relation': 15, 'object_part_of_series_type': 16, 'party_size_number': 17, 'served_dish': 18, 'rating_value': 19, 'facility': 20, 'playlist': 21, 'condition_temperature': 22, 'state': 23, 'year': 24, 'genre': 25, 'service': 26, 'condition_description': 27, 'movie_type': 28, 'timeRange': 29, 'cuisine': 30, 'object_name': 31, 'best_rating': 32, 'artist': 33, 'music_item': 34, 'party_size_description': 35, 'location_name': 36, 'geographic_poi': 37, 'country': 38}
        

    # all_slots = []
    # with open(trueLabelFile, 'r', encoding='utf8')as fin:
    #     for line in fin:
    #         sent = line.strip().split()
    #         for w in sent:
    #             if w.startswith("B"):
    #                 slot = w.split("-")[1]
    #                 if slot not in all_slots:
    #                     all_slots.append(slot)

    # labels2idx = {x:i for i, x in enumerate(all_slots)}
    # print(all_slots)
    # print(len(all_slots))

    if not os.path.exists(macro_f1_save_dir):
        os.makedirs(macro_f1_save_dir)

    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    
    labels2idx = gen_label2idx_by_descending_order(slots_stats_file)
    getResult(data_type)

# print(debug)
# print(epoch)
# if data_type == 'current_train_set':
#     print('type 1')
# elif data_type == 'current_predict':
#     print('type 2')
# elif data_type == "test":
#     print('type 3')
# else: print('Error')
