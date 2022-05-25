### in construction...

## 模型训练说明

### 1. 配置文件目录   

   `configs/`  
    &nbsp;&nbsp;&nbsp;|-- `by_count/`  存储采样方式为by_count的配置文件  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--   `classicSelfTraining_snips_by_count_10_bert.config` snips数据集上，每类slot采样10个样本，使用bert做embedder的设定 下的配置文件  
    &nbsp;&nbsp;&nbsp;|-- `by_ratio/`  存储采样方式为by_ratio的配置文件

注意事项

- 所有配置信息从配置文件中改
- 每次运行某个config需要更改`data_path`, `root_dir`, `serialization_dir`等参数，使得训练过程中生成的文件保存到指定的文件中。一般改bebug后的数字就行了，表示是第几次调试的结果。
- 当sample_count数值较小时，不要设置过大的batch_size(<=64), 实践证明模型可能会啥也学不到（还没想明白为什么会这样）


### 2. 训练脚本 `train.sh`

运行命令 
`nohup sh -x train.sh 1>logs/snips_count10_bert_debug33.log 2>&1 &`  

file_name是给输出日志起的名字，建议加上一些关键信息。具体使用参照该脚本的注释。


### 3. 结果可视化
程序  `results/result_analysis.py` 改完参数后直接运行。
具体使用参照该脚本的注释。图片保存在results/figures/{dataset_name}下。

