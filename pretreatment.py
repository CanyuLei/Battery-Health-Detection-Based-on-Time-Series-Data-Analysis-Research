import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from configs import FEATURE_NAMES, TARGET_NAME, SAMPLE_LENGTH, DATA_DIR, FEATURE_SCALER_PATH, TARGET_SCALER_PATH
from utils import load_pkl, save_pkl


def standard_scaler(values, scaler_path, mode="train"):
    # 标准化
    if mode == "train":
        scaler = StandardScaler()  # 定义标准化模型
        scaler.fit(values)  # 训练
        save_pkl(scaler_path, scaler)  # 保存
    else:
        scaler = load_pkl(scaler_path)  # 加载模型
    return scaler.transform(values)  # 转换


def standard_scaler_inverse(values, scaler_path):
    # 反标准化
    scaler = load_pkl(scaler_path)  # 加载模型
    return scaler.inverse_transform(values)  # 转换


def load_data():
    # 加载数据
    X = []  # 特征/输入
    y = []  # 目标值/输出
    for filename in tqdm(
        [filename for filename in os.listdir(DATA_DIR) if not filename.startswith("~") and filename.endswith(".xlsx")],
        desc="Loading data",
    ):  # 遍历文件夹下的所有文件
        data = pd.read_excel(os.path.join(DATA_DIR, filename))  # 读取文件
        for battery_id, battery_data in data.groupby("Cell No."):  # 按照电池编号分组
            if len(battery_data) <= SAMPLE_LENGTH:  # 如果电池数据长度小于样本长度
                continue  # 跳过
            X.append(battery_data[FEATURE_NAMES].values.tolist()[:SAMPLE_LENGTH])  # 前SAMPLE_LENGTH行 特征
            y.append(battery_data[TARGET_NAME].values.tolist()[SAMPLE_LENGTH - 1])  # 第SAMPLE_LENGTH行 目标值

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分训练集和验证集

    dim_0, dim_1, dim_2 = np.array(X_train).shape  # 维度
    #print(dim_0,dim_1,dim_2) = 4086 64 2
    X_train = (
        standard_scaler(np.array(X_train).reshape(dim_0 * dim_1, dim_2), FEATURE_SCALER_PATH, mode="train")
        .reshape(dim_0, dim_1, dim_2)
        .tolist()
    )  # 特征标准化 训练集训练

    dim_0, dim_1, dim_2 = np.array(X_val).shape  # 维度
    #print(dim_0, dim_1, dim_2) = 1022 64 2
    X_val = (
        standard_scaler(np.array(X_val).reshape(dim_0 * dim_1, dim_2), FEATURE_SCALER_PATH, mode="val")
        .reshape(dim_0, dim_1, dim_2)
        .tolist()
    )  # 特征标准化 验证集应用

    #dim_0, dim_1 = np.array(y_val).shape
    #print(dim_0,dim_1) = 1022 1
    y_train = standard_scaler(np.array(y_train), TARGET_SCALER_PATH, mode="train").tolist()  # 目标值标准化 训练集训练
    y_val = standard_scaler(np.array(y_val), TARGET_SCALER_PATH, mode="val").tolist()  # 目标值标准化 验证集应用


    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    load_data()
