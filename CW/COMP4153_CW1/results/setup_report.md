# 项目初始化检查记录

检查日期：2026-11-03。

## 环境

- 解释器：`C:/APP/Anaconda/envs/py3/python.exe`，Python 3.11.17。
- RTX 5060 Laptop GPU，约 8 GB 显存；驱动 591.97。
- 保留现有 torch 2.14.1+cu132 / torchvision 0.29.1+cu132，GPU 前向和反向运算通过。
- nvidia-smi 显示的 CUDA 13.1 与 torch 自带运行时 13.2 是不同信息；本机实际计算检查通过，无需据此重装工具链。
- 已补齐 SetFit、Sentence Transformers、Transformers、Datasets、Accelerate、Hugging Face Hub、Evaluate、Seaborn 等。JupyterLab、ipykernel 已存在。
- 核心包导入通过，pip check 无损坏依赖。安装过程中解析器将 fsspec 从 2026.9.0 调整为 2026.7.0。
- 小型离线 SetFit 检查通过：随机初始化微型 BERT、GPU 一步对比训练、分类头拟合、预测。它不是正式实验，也没有验证外部预训练模型下载或完整训练显存。
- 未安装 NLTK、spaCy、torchaudio：当前 SetFit 意图分类流程不依赖它们。
- 台式机未连接，尚未验证 RTX 4070 环境。

## 数据

扫描当前 ChatGPT 工作目录、课程 NLP 目录和默认 Hugging Face 缓存位置，未发现已有数据；未扫描全部磁盘。

| 数据 | 划分与条数 | 类别 |
|---|---|---|
| BANKING77 | train 10003 / test 3080 | 77 |
| CLINC150 域内 | train 15000 / val 3000 / test 4500 | 150 |
| CLINC OOS，单独保留 | train 100 / val 100 / test 1000 | OOS |

来源：
- https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data
- https://github.com/clinc/oos-eval/tree/master/data

下载脚本验证数量、字段、类别，并在 data/manifest.json 保存来源 URL、文件 SHA256 和检查时间。再次执行已验证为 REUSED，未重新下载。哈希用于记录本地版本，不是与官方发布校验值比对。

## 使用

在 Anaconda Prompt 执行：

```text
conda activate py3
cd /d C:\study\UNNC\PHD\PHD1\NLP\CW\COMP4153_CW1
python tools/check_environment.py
python tools/smoke_setfit.py
python tools/prepare_data.py
python -m jupyterlab
```

Notebook 00 是环境检查，01/02 仅为提纲；src/ 三个模块为占位文件。尚未进行课程正式实验。

最终 ZIP 仅打包符合要求的、独立运行并保留输出的实验 Notebook；不要直接打包此开发目录。AI 辅助包括本次环境配置、文件结构和调试，应如实纳入使用声明。
