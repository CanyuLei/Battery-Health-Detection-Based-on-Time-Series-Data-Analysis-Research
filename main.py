import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import (
    FEATURE_NAMES,
    TARGET_SCALER_PATH,
    MODEL_PATH,
    RESIDUAL_VISUALIZATION_PATH,
    COMPARISON_VISUALIZATION_PATH,
    RESULT_VALUES_PATH,
    DEVICE,
    CONFIG,
)
from datasets import Datasets
from models.transformer_model import Model
from pretreatment import standard_scaler_inverse, load_data
from train import train
from utils import regression_evaluate, residual_visualization, comparison_visualization


def train_run(X_train, X_val, y_train, y_val):
    train_datasets = Datasets(X_train, y_train)  # 训练数据集
    val_datasets = Datasets(X_val, y_val)  # 验证数据集

    train_loader = DataLoader(
        train_datasets,
        batch_size=CONFIG["batch_size"],#batch_size=128
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 训练数据加载器
    val_loader = DataLoader(
        val_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 验证数据加载器

    model = Model(inputs_size=len(FEATURE_NAMES), outputs_size=1).to(DEVICE)  # 模型
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])  # 优化器
    criterion = torch.nn.MSELoss()  # 损失函数

    train(train_loader, val_loader, model, optimizer, criterion)  # 开始训练


def test_run(X_test, y_test):
    test_datasets = Datasets(X_test, y_test)  # 测试数据集

    test_loader = DataLoader(
        test_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )  # 测试数据加载器

    model = Model(inputs_size=len(FEATURE_NAMES), outputs_size=1)  # 定义模型
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # 加载模型参数
    model.to(DEVICE)
    model.eval()  # 验证模式

    real_sets = []  # 真实值
    pred_sets = []  # 预测值
    for idx, batch_data in enumerate(test_loader):  # 遍历
        inputs, targets = batch_data  # 输入 输出
        outputs = model(inputs.to(DEVICE))  # 前向传播

        real_sets.extend(targets.numpy().tolist())  # 记录真实值
        pred_sets.extend(outputs.detach().cpu().numpy().tolist())  # 记录预测值

    real_sets = standard_scaler_inverse(np.array(real_sets), TARGET_SCALER_PATH).reshape(-1).tolist()  # 反归一化
    pred_sets = standard_scaler_inverse(np.array(pred_sets), TARGET_SCALER_PATH).reshape(-1).tolist()  # 反归一化

    regression_evaluate(real_sets, pred_sets)  # 打印模型的性能指标
    residual_visualization(real_sets, pred_sets, RESIDUAL_VISUALIZATION_PATH, fitting=True)  # 残差图
    comparison_visualization(real_sets, pred_sets, COMPARISON_VISUALIZATION_PATH)  # 预测图
    pd.DataFrame({"real values": real_sets, "pred values": pred_sets}).to_csv(RESULT_VALUES_PATH)  # 保存预测值和真实值


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()  # 加载数据
    train_run(X_train, X_val, y_train, y_val)  # 训练模型
    test_run(X_val, y_val)  # 测试模型并画图
