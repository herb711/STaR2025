# STaR

**Selective Token-aware Retrieval for Dynamic RAG (Retrieval Augmented Generation)**

---

## 📑 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [系统要求](#系统要求)
- [安装部署](#安装部署)
- [数据准备](#数据准备)
- [配置说明](#配置说明)
- [运行指南](#运行指南)
- [结果评估](#结果评估)

---

## 🧭 概述

STaR（Selective Token-aware Retrieval）是一个动态检索增强生成（RAG）框架，通过智能决策检索时机和内容，显著提升大语言模型的生成质量和准确性。

### 核心组件

- **CIND (Centrality-based Information Needs Detection)：** 基于中心性的信息需求检测，通过评估 LLM 的不确定性、token 重要性和语义意义，动态判断最佳检索时机
- **CQG (Centrality-enhanced Query Generation)：** 中心性增强的查询生成，结合中心性权重和上下文信息，生成精准的检索查询

---

## 🚀 核心特性

- **🎯 动态检索：** 根据 LLM 实时信息需求智能决定检索时机和内容
- **⚡ 轻量级集成：** 无缝集成任何基于 Transformer 的 LLM，无需额外训练或微调
- **📈 质量提升：** 生成更加信息丰富、上下文相关且连贯的文本
- **🛡️ 幻觉控制：** 通过中心性融合和一致性惩罚机制，显著降低幻觉生成
- **🔧 配置灵活：** 支持多种参数调节，适应不同应用场景

---

## 💻 系统要求

- **操作系统：** Linux / macOS / Windows
- **Python：** 3.9+
- **GPU：** 推荐 NVIDIA GPU（8GB+ 显存）
- **内存：** 16GB+ RAM
- **存储：** 50GB+ 可用空间（用于模型和数据集）

---

## 🛠️ 安装部署

### 1. 创建虚拟环境

```bash
conda create -n star python=3.9
conda activate star
```

### 2. 安装依赖

```bash
# 安装 PyTorch
pip install torch==2.1.1

# 安装项目依赖
pip install -r requirements.txt

# 下载语言模型
python -m spacy download en_core_web_sm
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import spacy; print('Spacy: OK')"
```

---

## 📊 数据准备

### 构建 Wikipedia 索引

#### 1. 下载 Wikipedia 数据

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
cd data/dpr
gzip -d psgs_w100.tsv.gz
cd ../..
```

#### 2. 部署 Elasticsearch

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz

# 启动 Elasticsearch
cd elasticsearch-7.17.9
nohup bin/elasticsearch &
cd ../..

# 构建索引
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

### 下载评估数据集

#### 2WikiMultihopQA
```bash
# 手动下载后解压到 data/2wikimultihopqa
# 下载地址：https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip
```

#### StrategyQA
```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip
```

#### HotpotQA
```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

#### IIRC
```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```

---

## ⚙️ 配置说明

### 核心参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model_name_or_path` | string | - | 预训练模型路径 |
| `method` | string | "STaR" | 使用的方法名称 |
| `dataset` | string | - | 数据集名称（hotpotqa/2wikimultihopqa/strategyqa/iirc） |
| `data_path` | string | - | 数据集路径 |
| `output_dir` | string | - | 结果输出目录 |

### 生成参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `generate_max_length` | int | 90 | 最大生成长度 |
| `fewshot` | int | 6 | Few-shot 示例数量 |
| `sample` | int | 1000 | 采样数量 |

### 检索参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `retriever` | string | "BM25" | 检索器类型 |
| `retrieve_topk` | int | 3 | 检索文档数量 |
| `retrieve_keep_top_k` | int | 25 | 保留的候选查询数量 |
| `query_formulation` | string | "real_words" | 查询构造方式 |

### STaR 特有参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `hallucination_threshold` | float | 0.15 | 幻觉检测阈值 |
| `halluc_gamma` | float | 0.3 | 幻觉调整参数 |
| `centrality_weight` | float | 0.15 | 中心性权重 |
| `consistency_penalty` | float | 5.0 | 一致性惩罚系数 |

### 配置文件模板

创建 `config/your_config.json`：

```json
{
    "model_name_or_path": "/path/to/your/model",
    "method": "STaR",
    "dataset": "hotpotqa",
    "data_path": "/path/to/data/hotpotqa",
    "fewshot": 6,
    "sample": 1000,
    "shuffle": false,
    "generate_max_length": 90,
    "query_formulation": "real_words",
    "retrieve_keep_top_k": 25,
    "output_dir": "../result/experiment_name",
    "retriever": "BM25",
    "retrieve_topk": 3,
    "hallucination_threshold": 0.15,
    "halluc_gamma": 0.3,
    "centrality_weight": 0.15,
    "consistency_penalty": 5.0,
    "check_real_words": true,
    "use_counter": true
}
```

---

## 🚀 运行指南

### 基础运行

#### 1. 进入工作目录
```bash
cd src
```

#### 2. 单次实验运行
```bash
python main.py --config ../config/your_config.json
```

#### 3. 批量实验运行
```bash
# 运行多个配置
for config in ../config/*.json; do
    echo "Running experiment with $config"
    python main.py --config "$config"
done
```

### 运行示例

#### HotpotQA 数据集
```bash
python main.py --config ../config/hotpotqa_llama2_7b.json
```

#### 2WikiMultihopQA 数据集
```bash
python main.py --config ../config/2wikimultihopqa_llama2_7b.json
```

#### 自定义参数运行
```bash
python main.py \
    --model_path "/path/to/model" \
    --dataset "hotpotqa" \
    --data_path "/path/to/data" \
    --output_dir "../result/custom_experiment" \
    --hallucination_threshold 0.2 \
    --centrality_weight 0.1
```

### 监控运行状态

#### 查看实时日志
```bash
tail -f ../result/experiment_name/*/output.log
```

#### 检查 GPU 使用情况
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

#### 检查 Elasticsearch 状态
```bash
curl -X GET "localhost:9200/_cluster/health?pretty"
```

### 常见问题解决

#### 1. 内存不足
```bash
# 减少批处理大小或样本数量
# 在配置文件中调整：
{
    "sample": 500,  // 减少样本数量
    "generate_max_length": 50  // 减少生成长度
}
```

#### 2. Elasticsearch 连接失败
```bash
# 重启 Elasticsearch
cd data/elasticsearch-7.17.9
bin/elasticsearch-service.sh restart
```

#### 3. 模型加载失败
```bash
# 检查模型路径和权限
ls -la /path/to/model
# 确保有足够的磁盘空间
df -h
```

---

## 📈 结果评估

### 自动评估

运行完成后，系统会自动在输出目录生成结果文件：

```bash
# 运行评估脚本
cd src
python evaluate.py --dir ../result/experiment_name/run_id
```

### 结果文件结构

```
result/
└── experiment_name/
    └── run_id/
        ├── config.json         # 实验配置
        ├── output.txt         # 原始输出结果
        ├── details.txt        # 评估详情
        ├── result.tsv         # 评估指标
        └── logs/              # 运行日志
```

### 输出文件说明

#### config.json
保存完整的实验配置，便于复现实验。

#### output.txt
包含每个问题的详细输出：
```json
{
    "qid": "问题ID",
    "prediction": "模型预测结果",
    "retrieve_count": 2,        // 检索次数
    "generate_count": 3,        // 生成轮数
    "hallucinated_count": 0,    // 幻觉检测次数
    "token_count": 64,          // 生成token数
    "sentence_count": 5         // 生成句子数
}
```

#### details.txt
包含评估详情：
```json
{
    "qid": "问题ID",
    "final_pred": "最终提取的答案",
    "EM": 1.0,                  // 精确匹配分数
    "F1": 0.95                  // F1分数
}
```

#### result.tsv
汇总评估指标：
```
Metric    Score
EM        0.6234
F1        0.7456
```

### 性能分析

#### 查看统计信息
```bash
# 查看检索统计
grep "retrieve_count" result/*/output.txt | python -c "
import sys, json
counts = [json.loads(line.split(':', 1)[1])['retrieve_count'] for line in sys.stdin]
print(f'平均检索次数: {sum(counts)/len(counts):.2f}')
print(f'最大检索次数: {max(counts)}')
"
```

#### 生成评估报告
```bash
python scripts/generate_report.py --result_dir ../result/experiment_name
```

### 对比实验

#### 批量评估多个实验
```bash
python scripts/batch_evaluate.py --result_dirs ../result/exp1 ../result/exp2 ../result/exp3
```

#### 生成对比图表
```bash
python scripts/plot_comparison.py --experiments exp1,exp2,exp3 --metrics EM,F1
```

---

## 📝 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

---

## 📧 联系

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

## 🔗 相关链接

- [论文地址](#)
- [数据集下载](#)
- [模型下载](#)
