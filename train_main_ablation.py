from trainer_ablation import run_training_loop, self_training_loop

# run_training_loop('./configs/baseline.json', if_predict= True)
# self_training_loop("./configs/classicSelfTraining.json")

import sys

# config_file = "./configs/by_count/classicSelfTraining_snips_by_count_10_bert.json"
# config_file = "./configs/test.json"
config_file = sys.argv[1]
self_training_loop(config_file)