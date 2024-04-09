import os
import random

import numpy as np
import torch


def makedir(_dir):
    # 新建文件夹
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def setup_seed(seed=42):
    # 随机因子
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# 特征和目标值
FEATURE_NAMES = ["Voltage", "Energy"]  # 特征名字列表 可增删改
#print(len(FEATURE_NAMES))
TARGET_NAME = ["Capacity"]  # 目标值名字

# 样本输入长度
SAMPLE_LENGTH = 64

# 根目录
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 模型
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
FEATURE_SCALER_PATH = os.path.join(OUTPUTS_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(OUTPUTS_DIR, "target_scaler.pkl")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")
R2_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "r2_visualization.png")
R2_VISUALIZATION_CSV_PATH = os.path.join(OUTPUTS_DIR, "r2_visualization.csv")
LOSS_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.png")
LOSS_VISUALIZATION_CSV_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.csv")
RESIDUAL_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "residual_visualization.png")
COMPARISON_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "comparison_visualization.png")
EVALUATE_RESULT_PATH = os.path.join(OUTPUTS_DIR, "evaluate_result.txt")
RESULT_VALUES_PATH = os.path.join(OUTPUTS_DIR, "result_values.csv")

# makedir
makedir(OUTPUTS_DIR)

# 模型覆盖
IS_COVER = True

# 随机种子 保证模型的可复现性
setup_seed(seed=42)

# 优先使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型相关配置
CONFIG = {
    "batch_size": 128,
    "lr": 0.001,
    "epoch": 500,
    "min_epoch": 20,
    "patience": 0.0002,
    "patience_num": 20,
}
