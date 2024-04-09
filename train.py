import os
import sys

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

from configs import (
    OUTPUTS_DIR,
    MODEL_PATH,
    R2_VISUALIZATION_PATH,
    R2_VISUALIZATION_CSV_PATH,
    LOSS_VISUALIZATION_PATH,
    LOSS_VISUALIZATION_CSV_PATH,
    IS_COVER,
    DEVICE,
    CONFIG,
)
from utils import epoch_visualization


def train_epoch(train_loader, model, optimizer, criterion, epoch):
    model.train()  # 训练模式
    real_sets = []  # 真实值
    pred_sets = []  # 预测值
    train_loss_records = []  # loss
    for idx, batch_data in enumerate(tqdm(train_loader, file=sys.stdout)):  # 遍历
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.to(DEVICE))  # 计算loss
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        real_sets.extend(targets.numpy().tolist())  # 记录真实值
        pred_sets.extend(outputs.detach().cpu().numpy().tolist())  # 记录预测值
        train_loss_records.append(loss.item())  # 记录loss

    train_mae = round(mean_absolute_error(real_sets, pred_sets), 4)  # 计算MAE
    train_r2 = round(r2_score(real_sets, pred_sets), 4)  # 计算R2
    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)  # 求loss均值
    print(f"[train] Epoch: {epoch} / {CONFIG['epoch']}, mae: {train_mae}, r2: {train_r2}, loss: {train_loss}")
    return train_mae, train_r2, train_loss


def evaluate(val_loader, model, criterion, epoch):
    model.eval()  # 验证模式
    real_sets = []  # 真实值
    pred_sets = []  # 预测值
    val_loss_records = []  # loss
    for idx, batch_data in enumerate(val_loader):  # 遍历
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.to(DEVICE))  # 计算loss

        real_sets.extend(targets.numpy().tolist())  # 记录真实值
        pred_sets.extend(outputs.detach().cpu().numpy().tolist())  # 记录预测值
        val_loss_records.append(loss.item())  # 记录loss

    val_mae = round(mean_absolute_error(real_sets, pred_sets), 4)  # 计算MAE
    val_r2 = round(r2_score(real_sets, pred_sets), 4)  # 计算R2
    val_loss = round(sum(val_loss_records) / len(val_loss_records), 4)  # 求loss均值
    print(f"[val]   Epoch: {epoch} / {CONFIG['epoch']}, mae: {val_mae}, r2: {val_r2}, loss: {val_loss}")
    return val_mae, val_r2, val_loss


def train(train_loader, val_loader, model, optimizer, criterion):
    best_val_r2 = -999  # 最好的r2
    patience_counter = 0  # 耐心度
    train_r2_records = []  # 训练r2
    train_loss_records = []  # 训练loss
    val_r2_records = []  # 验证r2
    val_loss_records = []  # 验证loss
    for epoch in range(1, CONFIG["epoch"] + 1):
        train_mae, train_r2, train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch)  # 训练
        val_mae, val_r2, val_loss = evaluate(val_loader, model, criterion, epoch)  # 验证

        train_r2_records.append(train_r2)  # 记录
        train_loss_records.append(train_loss)  # 记录
        val_r2_records.append(val_r2)  # 记录
        val_loss_records.append(val_loss)  # 记录

        if val_r2 - best_val_r2 > CONFIG["patience"]:  # 如果比之前模型效果好
            best_val_r2 = val_r2
            torch.save(
                model.state_dict(),
                MODEL_PATH
                if IS_COVER
                else os.path.join(OUTPUTS_DIR, f"{epoch}-train_mae{train_mae}-val_mae{val_mae}-model.pkl"),
            )  # 保存模型
            patience_counter = 0
        else:
            patience_counter += 1
        print(f"best current val r2: {best_val_r2}")

        if (patience_counter >= CONFIG["patience_num"] and epoch > CONFIG["min_epoch"]) or epoch == CONFIG["epoch"]:
            print(f"best val r2: {best_val_r2}, training finished!")
            break

    epoch_visualization(train_r2_records, val_r2_records, "R2", R2_VISUALIZATION_PATH)  # 绘制r2图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_r2_records) + 1)),
            "train loss": train_r2_records,
            "val loss": val_r2_records,
        }
    ).to_csv(
        R2_VISUALIZATION_CSV_PATH, index=False
    )  # 保存r2数据

    epoch_visualization(train_loss_records, val_loss_records, "loss", LOSS_VISUALIZATION_PATH)  # 绘制loss图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss_records) + 1)),
            "train loss": train_loss_records,
            "val loss": val_loss_records,
        }
    ).to_csv(
        LOSS_VISUALIZATION_CSV_PATH, index=False
    )  # 保存loss数据
