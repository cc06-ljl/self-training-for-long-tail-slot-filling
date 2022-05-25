#!/bin/bash

# Python解释器
PYTHON_BIN="/home/xcc/.conda/envs/py37/bin/python"
#PYTHON_BIN="/mnt/2/ljc/env/sslsf/bin/python"

# --------- 如果G4上没有其他也很占内存的程序在跑（如果有，可能会使机器超负荷运行），可以开多个进程同时跑config_dir文件夹下所有配置文件  -----------------

# 配置文件目录
# config_dir="configs/by_count"
# # 执行Python程序
# for file in `ls ${config_dir}`
# do
#     config_file=${config_dir}/${file}
#     echo "执行文件${config_file}..."
#     # nohup ${PYTHON_BIN} train_main.py ${config_file} &        
#     ${PYTHON_BIN} train_main.py ${config_file}
# done
# wait

# ---------- n个配置依次跑, 跑哪个配置文件就传它的相对路径。下面两条只是示例，可增可删 ----------------
# ${PYTHON_BIN} train_main.py configs/by_count/classicSelfTraining_snips_by_count_20.json
# ${PYTHON_BIN} train_main.py configs/by_count/classicSelfTraining_snips_by_count_10_bert.json
${PYTHON_BIN} train_main.py configs/by_count/classicSelfTraining_snips_by_count_10_bert_copy.json
# ${PYTHON_BIN} train_main.py configs/by_count/classicSelfTraining_few-nerd_by_count_10_seed_9_bert.json
# ----------- 以上两段，二选一 ----------------------

wait

echo "Done"