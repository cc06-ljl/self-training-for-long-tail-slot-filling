#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# here put the import lib
from unicodedata import digit
import encodings
import shutil
from shutil import copyfile
from random import sample
from typing import Any, Dict, Iterable, Tuple
import logging, os
import json
from datetime import datetime
import numpy as np

import torch
from allennlp.common.checks import check_for_gpu
from torch import serialization

from seqeval.metrics import classification_report, f1_score

# allennlp 日志部分，没有解释具体的作用，后续版本可以继承
if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)
logger = logging.getLogger(__name__)


from allennlp.common import Params
from allennlp.common.logging import prepare_global_logging
from allennlp.common.util import prepare_environment, dump_metrics
from allennlp.training.util import HasBeenWarned, create_serialization_dir
# from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary, AllennlpDataset
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader

from allennlp.models import Model
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from model import SlotTaggingModel
from dataset_reader import SlotFillingDatasetReader
from utils import SlotTaggingPredictor, calculate_resample_prob, calculate_sent_confidence, stat_slot_nums, calculate_slot_weight


def build_dataset_reader(params: Params) -> DatasetReader:
    return DatasetReader.from_params(params["dataset_reader"])


def read_data(reader: DatasetReader, params: Params):
    logging.info("Reading data")
    training_data = list(reader.read(params.pop("train_data_path", None)))
    validation_data = list(reader.read(params.pop("validation_data_path", None)))
    test_data = list(reader.read(params.pop("test_data_path", None)))
    return training_data, validation_data, test_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    logging.info("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(params: Params, vocab: Vocabulary, cls_num_list, slot_weight) -> Model:
    logging.info("Building the model")
    loss_params = params.as_dict()["hyper-params"]["loss"]
    return Model.from_params(params.pop("model", None), **loss_params, vocab=vocab, cls_num_list=cls_num_list, slot_weight=slot_weight)


def build_data_loaders(train_data, dev_data, test_data, params):
 
    batch_size = params["train_data_loader"].pop("batch_size", 128)
    shuffle = params["train_data_loader"].pop("shuffle", True)
    
    train_loader = SimpleDataLoader(train_data, batch_size, shuffle=shuffle)
    dev_loader = SimpleDataLoader(dev_data, batch_size, shuffle=shuffle)
    
    batch_size = params["test_data_loader"].pop("batch_size", 128)
    shuffle = params["test_data_loader"].pop("shuffle", False)

    test_loader = SimpleDataLoader(test_data, batch_size, shuffle=shuffle)
    return train_loader, dev_loader, test_loader


def build_trainer(
        params: Params,
        model: Model,
        train_loader: DataLoader,
        dev_loader: DataLoader
) -> Trainer:
    trainer = Trainer.from_params(params.pop("trainer", None), model=model, data_loader=train_loader,
                                  validation_data_loader=dev_loader)
    return trainer
    

def evaluate_testset(serialization_dir: str, model: Model, test_loader: DataLoader, cuda_device: int = -1, mode: str = "test"):
    logging.info(f"Evaluating on the {mode} set")
    import torch  # import here to ensure the republication of the experiment
    model.load_state_dict(torch.load(os.path.join(serialization_dir, "best.th")))
    test_metrics = evaluate(model, test_loader, cuda_device=cuda_device)
    logging.info(f"Metrics on the  {mode} set: {test_metrics}")
    return test_metrics


def run_training_loop(config_file_path: str, slot_label_list:list, if_test: bool=True, if_predict: bool=False, epoch=None):
    # read config file
    params = Params.from_file(config_file_path)
    # get the value of `serialization_dir`
    serialization_dir = params.as_dict()["trainer"]["serialization_dir"]
    # print(serialization_dir)
    # save configs in serialization_dir
    params.to_file(os.path.join(serialization_dir, "config.json"))
    # creates the serialization directory for recording this epoch
    create_serialization_dir(params, serialization_dir, recover=True, force=False)

    prepare_global_logging(serialization_dir=serialization_dir)
    # prepare_global_logging(serialization_dir=serialization_dir, file_friendly_logging=True)
    prepare_environment(params)  # 设置random seed

    train_data_path = params.as_dict()["train_data_path"]
    dev_data_path = params.as_dict()["validation_data_path"]
    test_data_path = params.as_dict()["test_data_path"]
    cuda_device = params.as_dict()["trainer"]["cuda_device"]

    # hyper-params
    resample_params_dict = params.as_dict()["hyper-params"]["resample"]
    # {
    #     "threshold": params.as_dict()["hyper-params"]["resample"]["threshold"],   # 伪标签选取策略中的 置信度阈值
    #     "mu": params.as_dict()["hyper-params"]["resample"]["mu"],           # 伪标签选取策略中 re-sampling(cReST)中的mu超参
    #     "pattern_id": params.as_dict()["hyper-params"]["resample"]["pattern_id"] # 对于新增槽位数目为0的处理，采用方式1还是2
    # }

    # prepare data -- vocab and loader 
    dataset_reader = build_dataset_reader(params)

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data, test_data = read_data(dataset_reader, params)
    vocab = build_vocab(train_data + dev_data + test_data)

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    train_loader, dev_loader, test_loader = build_data_loaders(train_data, dev_data, test_data, params)
    
    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    test_loader.index_with(vocab)

    # use params and vocab to build model
    # model = build_model(params, vocab)

    # -------------- ccadd: 为迎合LDAM损失定义，给模型初始化时传入类别数量 【0315】-----------
    slot2idx = vocab._token_to_index['labels']
    slot_num_dict = stat_slot_nums(train_data_path, mode="all")   # 在损失函数中, mode应该用all吧，利用所有累积的样本
    slot_weight_dict = calculate_slot_weight(slot_num_dict, resample_params_dict["mu"], resample_params_dict["pattern_id"], slot_label_list)

    slots = sorted(slot2idx.items(), key=lambda x:x[1])
    cls_num_list = []
    slot_weight = []
    O_index = 0   # 记录O_index的前缀
    for i, (s, _) in enumerate(slots):
        if s == "O":
            O_index = i
            continue
        cls_num_list.append(slot_num_dict[s[2:]])
        slot_weight.append(slot_weight_dict[s[2:]])

    cls_num_list.insert(O_index, max(cls_num_list))    # O的数量默认等于最多的slot数量. 更合理的做法???
    slot_weight.insert(O_index, min(slot_weight))    # O的weight默认等于最小的slot weight. 更合理的做法???
    slot_weight = torch.FloatTensor(slot_weight).cuda(cuda_device)

    model = build_model(params, vocab, cls_num_list, slot_weight)
    # --------------------------------


    # use params, model and data_loader to build trainer
    trainer = build_trainer(params, model, train_loader, dev_loader)   # trainer.cuda_device 查看模型在哪块设备上

    logging.info("Starting training")
    trainer.train()
    logging.info("Finished training")

    dev_metrics = evaluate_testset(serialization_dir, model, dev_loader, cuda_device, mode="dev")

    # evaluate_testset(serialization_dir, model, test_loader, cuda_device, mode="test")

    dev_metrics['macro_f1'] = cal_dev_macrof1(serialization_dir, dev_data_path, cuda_device)

    with open(os.path.join(serialization_dir, "dev.json"), "w", encoding="utf8")as fout:
        json.dump(dev_metrics, fout, indent=4)

    
    if if_predict:
        unlabeled_data_path = params.as_dict()["unlabeled_data_path"]
        save_data_path = os.path.join(params.as_dict()["data_path"], f"epoch_{epoch+1}")
        test_result_save_data_path = os.path.join(params.as_dict()["data_path"], f"epoch_{epoch}")   # xcc add :把本轮测试集的结果存在本轮的saved_data/epoch_*下
        print('************************************************************************')
        print(serialization_dir, unlabeled_data_path, save_data_path)
        print('************************************************************************')
        run_predict_persudo_label(serialization_dir, cuda_device, unlabeled_data_path, save_data_path, train_data_path, resample_params_dict, slot_label_list)
        analysis_test_result(serialization_dir, cuda_device, test_result_save_data_path, test_data_path)

    # return dev_metrics['f1']
    return dev_metrics['macro_f1']


def cal_dev_macrof1(result_path, dev_data_path, cuda_device):
    archive = load_archive(result_path, cuda_device=cuda_device)    # XCC 添加参数cuda_device;
    predictor = Predictor.from_archive(archive=archive, predictor_name="slot_tagging_predictor")
    with open(os.path.join(dev_data_path,'seq.in'), 'r', encoding='utf-8')as fin:
        test_data = fin.readlines()
    with open(os.path.join(dev_data_path,'seq.out'), 'r', encoding='utf-8')as fin:
        test_label = fin.readlines()
    
     # xcc add: 把全部数据一次性加载进去会CUDA out of memory --> 所以改为批量预测
    all_test_data_with_true_label = []
    batch_test_data = []
    for text, labels in zip(test_data, test_label):
        if len(test_data) == 128:
            all_test_data_with_true_label.append(batch_test_data)
            batch_test_data = []

        data = {}
        data["tokens"] = text.strip().split()
        data["true_labels"] = labels.strip().split()
        batch_test_data.append(data)
    
    if len(batch_test_data) != 0:
        all_test_data_with_true_label.append(batch_test_data)
    
    y_pred = []
    y_true = []
    for batch in all_test_data_with_true_label:
        batch_outputs = predictor.predict(batch)
        for output in batch_outputs:
            y_pred.append(output['predict_labels'])
            y_true.append(output['true_labels'])
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')   
    return macro_f1


def analysis_test_result(result_path, cuda_device, save_data_path, test_data_path):
    archive = load_archive(result_path, cuda_device=cuda_device)    # XCC 添加参数cuda_device;
    print("predicting...")
    predictor = Predictor.from_archive(archive=archive, predictor_name="slot_tagging_predictor")
    with open(os.path.join(test_data_path,'seq.in'), 'r', encoding='utf-8')as fin:
        test_data = fin.readlines()
    with open(os.path.join(test_data_path,'seq.out'), 'r', encoding='utf-8')as fin:
        test_label = fin.readlines()
    

    # xcc add: 把全部数据一次性加载进去会CUDA out of memory --> 所以改为批量预测
    all_test_data_with_true_label = []
    batch_test_data = []
    for text, labels in zip(test_data, test_label):
        if len(batch_test_data) == 128:
            all_test_data_with_true_label.append(batch_test_data)
            batch_test_data = []

        data = {}
        data["tokens"] = text.strip().split()
        data["true_labels"] = labels.strip().split()
        batch_test_data.append(data)
    
    if len(batch_test_data) != 0:
        all_test_data_with_true_label.append(batch_test_data)
    
    output_dicts = []
    for batch in all_test_data_with_true_label:
        output_dicts.extend(predictor.predict(batch))

    if not os.path.exists(os.path.join(save_data_path, "test")):
        os.makedirs(os.path.join(save_data_path, "test"))

    all_seq_prediction = []
    all_seq_truelabel = []

    with open(os.path.join(save_data_path, "test", "seq.in"), 'w', encoding='utf8') as seqin, \
        open(os.path.join(save_data_path, "test", "seq.out"), 'w', encoding='utf8') as seqout, \
        open(os.path.join(save_data_path, "test", "true.label"), 'w', encoding='utf8') as labelout:
        for output in output_dicts:
            seqin.write(" ".join(output['tokens'])+'\n')
            seqout.write(" ".join(output['predict_labels'])+'\n')
            labelout.write(" ".join(output['true_labels'])+'\n')
            all_seq_prediction.append(output['predict_labels'])
            all_seq_truelabel.append(output['true_labels'])

    # --- 计算test macro f1
    assert len(all_seq_prediction) == len(all_seq_truelabel), "预测和真实标签数量对不齐"

    metric_report = classification_report(all_seq_truelabel, all_seq_prediction, digits=4)
    with open(os.path.join(save_data_path, "test_classification_report.txt"), 'w', encoding='utf8') as f:
        f.write(metric_report)


def run_predict_persudo_label(result_path, cuda_device, unlabeled_data_path, save_data_path, train_data_path, resample_params_dict, slot_label_list=None, pattern_id=2):
    """ 
        每次迭代用在labelled data训练好的模型伪注释unlabelled data,把高置信度(大于门限值)的unlabelled sentence-pseudo-label 加入到原来的labelled data,
        同时把他们从unlablled data中删除。在下一次迭代中用新的labelled data训练
    """
    #need to add the fuction for saving peasudo-labeled data
    archive = load_archive(result_path, cuda_device=cuda_device)    # XCC 添加参数cuda_device;  next(archive.model.parameters()).device 查看在哪块设备上 
    print("predicting...")
    predictor = Predictor.from_archive(archive=archive, predictor_name="slot_tagging_predictor")   # next(archive.model.parameters()).device

    threshold = resample_params_dict["threshold"]
    mu = resample_params_dict["mu"]
    pattern_id = resample_params_dict["pattern_id"]
    alpha = resample_params_dict["alpha"]
    beta = resample_params_dict["beta"]
    gama = resample_params_dict["gama"]
    resample_mode = resample_params_dict["resample_mode"]
    

    with open(os.path.join(unlabeled_data_path,'seq.in'), 'r', encoding='utf-8')as fin:
        unlabel_data = fin.readlines()       # 待优化 —— 在文件很大的情况下, 一次性读取文件中所有内容，容易造成内存爆炸.
    with open(os.path.join(unlabeled_data_path,'seq.out'), 'r', encoding='utf-8')as fin:
        unlabel_label = fin.readlines()
    # unlabel_data = [{"tokens": text.strip().split()} for text in unlabel_data[:10] if len(text) > 0]

    all_unlabel_data_with_true_label = []
    batch_unlabel_data = []
    for text, labels in zip(unlabel_data, unlabel_label):
        if len(batch_unlabel_data) == 128:
            all_unlabel_data_with_true_label.append(batch_unlabel_data)
            batch_unlabel_data = []

        data = {}
        data["tokens"] = text.strip().split()
        data["true_labels"] = labels.strip().split()
        batch_unlabel_data.append(data)
    
    if len(batch_unlabel_data) != 0:
        all_unlabel_data_with_true_label.append(batch_unlabel_data)

    output_dicts = []
    for batch in all_unlabel_data_with_true_label:
        output_dicts.extend(predictor.predict(batch))

    # output_dicts = predictor.predict(unlabel_data_with_true_label[10:20])
    # predict_labels = output_dict['predict_labels']

        
    if not os.path.exists(os.path.join(save_data_path, "labeled")):
        os.makedirs(os.path.join(save_data_path, "labeled"))
    if not os.path.exists(os.path.join(save_data_path, "unlabeled")):
        os.makedirs(os.path.join(save_data_path, "unlabeled"))

    # ------------------ New add ----------------------
    # 获取本轮新加入的pseudo-labeled data的槽位数量信息. 注意，第一个epoch就用初始的所有训练数据 

    # TODO：用新加入的计算slot-span-level weight 【03-03 Done】
    # 但还需要对某些slot新加入的数据量为0的情况做处理 【03-04 Done】

    labeled_data_path  = unlabeled_data_path.replace("unlabeled", "labeled")
    if labeled_data_path.split("/")[-2] == "epoch_1":
        slot_num_dict = stat_slot_nums(labeled_data_path, mode="all")
    else:
        slot_num_dict = stat_slot_nums(labeled_data_path, mode="add")
    slot_weight_dict = calculate_slot_weight(slot_num_dict, mu, pattern_id, slot_label_list, save_data_path)
    # --------------------------------------------------

    new_labeled_data = []
    new_unlabeled_data = []
    for output in output_dicts:
        logits = output["tag_logits"]
        
        mask = torch.tensor(output["mask"]).sum()
        logits = torch.softmax(torch.tensor(logits)[:mask],dim=1)
        max_score, _= torch.max(logits, dim=1)

# ================ 新策略【0417】 =====================
        # entity_slot_cnt = 0
        # weight_sum = 0.0
        # sent_confidence = calculate_sent_confidence(slot_weight_dict, output['predict_labels'], max_score, gama=gama)
        
        # if sent_confidence < threshold:
        #     new_unlabeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
        # else:
        #     resamle_prob = calculate_resample_prob(sent_confidence, threshold, alpha=alpha, beta=beta, mode=resample_mode)
            
        #     for p in output['predict_labels']:
        #         if p.startswith('B') or p.startswith('I'):
        #             weight_sum += slot_weight_dict[p[2:]] 
        #             entity_slot_cnt += 1 

        #     binomial_p = (weight_sum / entity_slot_cnt) * resamle_prob

        #     if binomial_p > 1 or binomial_p < 0:
        #         logging.info(f"!!!!!!! ValueError! 伯努利分布的p值计算不符合要求! p={binomial_p}, sent_confidence={sent_confidence}, resamle_prob={resamle_prob}") 
        #         new_unlabeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
        #         continue

        #     if (np.random.binomial(1, binomial_p, 1)[0]):
        #         new_labeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
        #     else:
        #         new_unlabeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))


# ================ 原策略 =========================
        # weight_sum = 0.0     # 这句话被选中的概率
        # score_sum = 0.0
        # B_I_num = 0.0

        # for i, p in enumerate(output['predict_labels']):    # 返回的预测标签p直接是标签,如I-artist
        #     if p.startswith('B') or p.startswith('I'):
        #         score_sum += max_score[i]
        #         weight_sum += slot_weight_dict[p[2:]]
        #         B_I_num += 1.0
        
        # mean_score = score_sum / (B_I_num + 0.000001)
        # mean_prob = weight_sum / (B_I_num + 0.000001)

        # if mean_score > threshold:                             # 每句话tokens的平均得分>threshold, 则加入labeled_data，否则仍然是unlabeled_data
        #     if np.random.binomial(1, mean_prob, 1)[0]:         # 伯努利分布，返回1（选中）的概率为mean_prob
        #     # if np.random.binomial(1, mean_prob, 1)[0] and np.random.binomial(1, 0.9, 1)[0]:         # -- 为了学得更缓慢,再过一层伯努利
        #         new_labeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
        #     else:
        #         new_unlabeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
        # else:
        #     new_unlabeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
# ===========================================================


# ================= baseline: original resample ================
        score_sum = 0.0
        B_I_num = 0.0

        for i, p in enumerate(output['predict_labels']):    # 返回的预测标签p直接是标签,如I-artist
            if p.startswith('B') or p.startswith('I'):
                score_sum += max_score[i]
                B_I_num += 1.0
        
        mean_score = score_sum / (B_I_num + 0.000001)

        if mean_score > threshold:                             # 每句话tokens的平均得分>threshold, 则加入labeled_data，否则仍然是unlabeled_data
            new_labeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
        else:
            new_unlabeled_data.append((output["tokens"], output["predict_labels"], output["true_labels"]))
    
# =================================================

    print('------------------------------------------')
    print("new labeled data length:",len(new_labeled_data))
    print("remaining unlabeled data length:",len(new_unlabeled_data))
    print('------------------------------------------')

    source_text_file = os.path.join(train_data_path, "seq.in")
    source_label_file = os.path.join(train_data_path, "seq.out")
    source_true_label_file = os.path.join(train_data_path, "true.label")
    target_text_file = os.path.join(save_data_path, "labeled", "seq.in")
    target_label_file = os.path.join(save_data_path, "labeled", "seq.out")
    target_true_label_file = os.path.join(save_data_path, "labeled", "true.label")
    
    copyfile(source_text_file, target_text_file)
    copyfile(source_label_file, target_label_file)
    copyfile(source_true_label_file, target_true_label_file)
    assert os.path.isfile(target_text_file)
    assert os.path.isfile(target_label_file)
    assert os.path.isfile(target_true_label_file)

    with open(os.path.join(save_data_path, "labeled", "seq.in"), 'a', encoding='utf8')as seqin:
        with open(os.path.join(save_data_path, "labeled", "seq.out"), 'a', encoding='utf8')as seqout:
            with open(os.path.join(save_data_path, "labeled", "true.label"), 'a', encoding='utf8')as labelout:
                for tokens, slots, true_label in new_labeled_data:
                    seqin.write(' '.join(tokens) + '\n')
                    seqout.write(' '.join(slots) + '\n')
                    labelout.write(' '.join(true_label) + '\n')

    # 新加入的伪标签数据
    with open(os.path.join(save_data_path, "labeled", "new_added_seq.in"), 'w', encoding='utf8') as seqin, \
        open(os.path.join(save_data_path, "labeled", "new_added_seq.out"), 'w', encoding='utf8')as seqout, \
        open(os.path.join(save_data_path, "labeled", "new_added_true.label"), 'w', encoding='utf8')as labelout:
                for tokens, slots, true_label in new_labeled_data:
                    seqin.write(' '.join(tokens) + '\n')
                    seqout.write(' '.join(slots) + '\n')
                    labelout.write(' '.join(true_label) + '\n')

    # with open(os.path.join(save_data_path, "unlabeled", "seq.in"), 'w', encoding="utf8")as seqin:
    #     with open(os.path.join(save_data_path, "unlabeled", "seq.out"), 'w', encoding="utf8")as seqout:
    #         with open(os.path.join(save_data_path, "labeled", "predictd.label"), 'w', encoding='utf8')as resout:
    #             for tokens, slots, true_label in new_unlabeled_data:
    #                 seqin.write(' '.join(tokens) + '\n')
    #                 seqout.write(' '.join(true_label) + '\n')
    #                 resout.write(' '.join(slots) + '\n')
    with open(os.path.join(save_data_path, "unlabeled", "seq.in"), 'w', encoding="utf8")as seqin, \
        open(os.path.join(save_data_path, "unlabeled", "seq.out"), 'w', encoding="utf8")as seqout, \
        open(os.path.join(save_data_path, "unlabeled", "predictd.label"), 'w', encoding='utf8') as resout:     # Xcc Fixed
                for tokens, slots, true_label in new_unlabeled_data:
                    seqin.write(' '.join(tokens) + '\n')
                    seqout.write(' '.join(true_label) + '\n')
                    resout.write(' '.join(slots) + '\n')


def make_ssl_data(params):
    # params = Params.from_file(config_file_path)
    save_data_path = params.as_dict()["data_path"]
    train_data_path = params.as_dict()["train_data_path"]
    random_seed = int(params.as_dict()["random_seed"])
    sample_mode = params.as_dict()["sampler"]["mode"]
    sample_type = params.as_dict()["sampler"]["type"]
    sample_ratio = params.as_dict()["sampler"]["sample_ratio"]
    sample_count = params.as_dict()["sampler"]["sample_count"]
    sampled_data, unsample_data = sampleData(train_data_path, sample_mode, sample_type, sample_ratio, sample_count, random_seed)
    init_data_path = os.path.join(save_data_path, "epoch_1")
    
    # if not os.path.exists():
    # print(os.path.join(init_data_path, "dataset", "labeled"))
    if not os.path.exists(os.path.join(init_data_path, "labeled")):
        os.makedirs(os.path.join(init_data_path, "labeled"))
    if not os.path.exists(os.path.join(init_data_path, "unlabeled")):
        os.makedirs(os.path.join(init_data_path, "unlabeled"))
    
    with open(os.path.join(init_data_path, "labeled", "seq.in"), 'w', encoding='utf8')as seqin:
        with open(os.path.join(init_data_path, "labeled", "seq.out"), 'w', encoding='utf8')as seqout:
            with open(os.path.join(init_data_path, "labeled", "true.label"), 'w', encoding='utf8')as trueout:
                for tokens, slots in sampled_data:
                    seqin.write(' '.join(tokens) + '\n')
                    seqout.write(' '.join(slots) + '\n')
                    trueout.write(' '.join(slots) + '\n')
                    
    with open(os.path.join(init_data_path,  "unlabeled", "seq.in"), 'w', encoding="utf8")as seqin:
        with open(os.path.join(init_data_path, "unlabeled", "seq.out"), 'w', encoding="utf8")as seqout:
            for tokens, slots in unsample_data:
                seqin.write(' '.join(tokens) + '\n')
                seqout.write(' '.join(slots) + '\n')
    

def sampleData(datapath, mode="coarse", stype="by_count", sample_ratio=0.1, sample_count=10, random_seed=1):
    logging.info("Sampling data...")
    start_time = datetime.now()

    samples = []
    seq_in_file = os.path.join(datapath, "seq.in")
    seq_out_file = os.path.join(datapath, "seq.out")
    
    # np.random.seed(random_seed)
    np.random.seed(9)
    with open(seq_in_file, "r") as f:
        seq_in_lines = f.readlines()
    with open(seq_out_file, "r") as f:
        seq_out_lines = f.readlines()
    assert len(seq_in_lines) == len(seq_out_lines)
    for seq_in_line, seq_out_line in zip(seq_in_lines, seq_out_lines):
        tokens = seq_in_line.strip().split(" ")
        tokens = [token for token in tokens if token]
        slots = seq_out_line.strip().split(" ")
        slots = [slot for slot in slots if slot]
        assert len(tokens) == len(slots)
        # if (tokens, slots) not in samples:
        #总共13084，非重复12860
        samples.append((tokens, slots))
    
    # print(len(samples))
    def parse_bio(tags):
        """
        给定BIO标注列表，返回一个list，包含这句话中的slot label及其对应长度。
        这里slot_label指：标签类别，不带前缀。如slot=I-round_trip, 则slot_label=round_trip.
        """
        result = []
        token_count = 0   # 用来记录除O以外的这个slot_label的槽位数，比如slots=[B-round_trip I-round_trip],则slot_label=round_trip, token_count=2.表示从此位起，有2个连续token(包含此位)都属于这次slot_label.
        last_tag = "O"
        current_label = None
        for tag in tags:
            if last_tag == "O":
                if tag != "O":
                    assert tag[0] == "B" or tag[0] == "I"
                    current_label = tag[2:]
                    token_count = 1
            else:
                if tag == "O":
                    result.append((current_label, token_count))
                    current_label = current_label = None
                    token_count = 0
                elif tag[0] == "":
                    token_count += 1
                else:
                    result.append((current_label, token_count))
                    current_label = tag[2:]
                    token_count = 1
            last_tag = tag
        if current_label is not None:
            result.append((current_label, token_count))
        return result

    all_slots = []   # 不带BIO前缀的标签集合
    slot2example = {}   # 每个标签的样本字典，key为标签，如round_trip，value为包含这个slot label的utterances[(tokens, slots),...]

    #***********************************************************************************
    for tokens, slots in samples:
        for slot_label, count in parse_bio(slots):
            if slot_label not in all_slots:
                all_slots.append(slot_label)
                slot2example[slot_label] = [(tokens, slots)]
            else:
                slot2example[slot_label].append((tokens, slots))
    sampled_data = []
    unsampled_data = []

    if mode == "coarse":
        # label_cnt_dict = {}
        for slot in all_slots:
            if stype == "by_ratio":
                sample_count = int(len(slot2example[slot]) * sample_ratio)   # 按比例sample_ratio确定每个标签类别要抽取的有监督样本数; 剩余的都是无监督的样本
            # sample_for_labeled = np.random.choice(len(slot2example[slot]), size=sample_count, replace=False)
            np.random.shuffle(slot2example[slot])
            # sample_by_slot = list(np.random.choice(slot2example[slot], size=sample_count, replace=False))
            num = 0
            for s in slot2example[slot]:
                if s not in sampled_data:
                    if num < sample_count:
                        sampled_data.append(s)
                        slot2example[slot].remove(s)
                        num += 1
                    else:
                        break
                # -------
                # if s not in temp_cnt_dict:
                #     temp_cnt_dict[s] = 0
                # temp_cnt_dict[s] += 1
                # if s 
                # -------
                    # elif s not in unsampled_data:
                        # unsampled_data.append(s)
        for slot in all_slots:
            # if stype == "by_ratio":
            #     sample_count = int(len(slot2example[slot]) * sample_ratio)
            
            # # sample_for_labeled = np.random.choice(len(slot2example[slot]), size=sample_count, replace=False)
            # np.random.shuffle(slot2example[slot])
            # sample_by_slot = list(np.random.choice(slot2example[slot], size=sample_count, replace=False))
            for s in slot2example[slot]:
                if s not in sampled_data and s not in unsampled_data:
                        unsampled_data.append(s)

    # print(sampled_data[0])
    # print(len(sampled_data))
    # # print(len(list(set(sampled_data))))
    # print(all_slots)
    # print(len(all_slots))
    # print(len(unsampled_data))
    # # print(len(slot2example))
    # print(len(samples))
    taken_time = datetime.now() - start_time
    logging.info(f"Finishing sampling data.Time cosuming: {taken_time}")

    return sampled_data, unsampled_data


def self_training_loop(init_config_file_path):
    # read the init_config_file
    init_params = Params.from_file(init_config_file_path)
    # this path should be the root floder for this whole training process
    init_serialization_dir = init_params.as_dict()["trainer"]["serialization_dir"]
    # serialization_dir必须是空文件夹，不然后面会报错
    if os.path.exists(init_serialization_dir):
        shutil.rmtree(init_serialization_dir)
    os.makedirs(init_serialization_dir)

    # 保存每一轮迭代的labeled data、unlabeled data、test data的目录, 
    init_data_path = init_params.as_dict()["data_path"]
    if not os.path.exists(init_data_path):
        os.makedirs(init_data_path)

    # 保存每一轮迭代的teacher model的训练过程信息，包括config、log、test_metric、best_model的目录，即每一轮serialization_dir的父目录
    root_path = init_params.as_dict()["root_dir"]

    # training params 
    best_performance = 0
    init_patience = 10
    patience = init_patience
    ST_epoch = 1
    # split training data for ssl slot-filling
    make_ssl_data(init_params)
    init_params.to_file(os.path.join(root_path, "config.json"))

    # Get label list
    slot_nums_dict = stat_slot_nums(os.path.join(init_data_path, "epoch_1/labeled"), mode="all")
    slot_label_list = list(slot_nums_dict.keys())

    #************************* loop the ST process *********************************
    while patience > 0:
        # make a new config for current epoch
        params = Params.from_file(os.path.join(root_path, "config.json"))
        # for each epoch, we should make a new directory for saving model, config and new data
        serialization_dir = os.path.join(root_path, f"res_of_epoch_{ST_epoch}")
        if not os.path.exists(serialization_dir):
            os.mkdir(serialization_dir)
        # read data for current epoch
        params["train_data_path"] = os.path.join(init_data_path, f"epoch_{ST_epoch}", "labeled")
        params["unlabeled_data_path"] = os.path.join(init_data_path, f"epoch_{ST_epoch}", "unlabeled")
        params["trainer"]["serialization_dir"] = serialization_dir
        params["trainer"]["callbacks"] = [{"type": "tensorboard", "serialization_dir": serialization_dir}]

        # save the config
        params.to_file(os.path.join(root_path, f"config_for_epoch_{ST_epoch}.json"))
        current_config_file_path = os.path.join(root_path, f"config_for_epoch_{ST_epoch}.json")

        # train teacher model, and use the trained model to predict peasudo label for unlabeled data
        # test_performance = run_training_loop(current_config_file_path, if_test=True, if_predict=True, epoch=ST_epoch)
        dev_performance = run_training_loop(current_config_file_path, slot_label_list, if_test=True, if_predict=True, epoch=ST_epoch)

        # 直到连续5次迭代模型性能没有提升,next才会不满足while循环条件
        if dev_performance > best_performance:
            best_performance = dev_performance
            patience = init_patience
        else: 
            patience -= 1
        ST_epoch += 1

    print("best performance of devset: ", best_performance)
        

if __name__ == "__main__":
    # init_params = Params.from_file("./configs/classicSelfTraining.json")
    # make_ssl_data(init_params)
    debug = 6
    epoch = 16
    
    result_path = f"results/self_train_debug_{debug}/res_of_epoch_{epoch}"
    unlabeled_data_path = f"results/self_train_debug_{debug}/saved_data/epoch_{epoch}/unlabeled"
    save_data_path = f"results/self_train_debug_{debug}/saved_data/epoch_{epoch+1}"
    train_data_path = f"results/self_train_debug_{debug}/saved_data/epoch_{epoch}/labeled"
    run_predict_persudo_label(result_path, unlabeled_data_path, save_data_path, train_data_path, threshold=0.9)
    
