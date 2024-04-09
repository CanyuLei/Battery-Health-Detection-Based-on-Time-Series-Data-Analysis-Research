import dill
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from configs import EVALUATE_RESULT_PATH

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def save_pkl(filepath, data):
    # 保存模型
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    # 加载模型
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def save_txt(filepath, data):
    # 保存txt
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def epoch_visualization(y1, y2, name, output_path):
    # epoch变化图
    plt.figure(figsize=(16, 9), dpi=100)  # 定义画布
    plt.plot(y1, marker=".", linestyle="-", linewidth=2, label=f"train {name}")  # 曲线
    plt.plot(y2, marker=".", linestyle="-", linewidth=2, label=f"val {name}")  # 曲线
    plt.title(f"训练过程中 {name} 变化图", fontsize=24)  # 标题
    plt.xlabel("epoch", fontsize=20)  # x轴标签
    plt.ylabel(name, fontsize=20)  # y轴标签
    plt.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    plt.legend(loc="best", prop={"size": 20})  # 图例
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示


def regression_evaluate(y_true, y_pred):
    # 回归模型的性能指标
    evaluate_result = ""
    evaluate_result += f"评估指标为:"
    evaluate_result += f"\nMAE: {round(mean_absolute_error(y_true, y_pred), 4)}"
    evaluate_result += f"\nRMSE: {round(pow(mean_squared_error(y_true, y_pred), 0.5), 4)}"
    evaluate_result += f"\nMAPE: {round(mean_absolute_percentage_error(y_true, y_pred), 4)}"
    evaluate_result += f"\nR2: {round(r2_score(y_true, y_pred), 4)}"
    print(evaluate_result)
    save_txt(EVALUATE_RESULT_PATH, evaluate_result)


def residual_visualization(y_real, y_pred, output_path, fitting=False):
    # 绘制预测值和真实值的对比图
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)

    ax.text(
        min(y_real),
        max(y_real),
        f"$MAE={round(mean_absolute_error(y_real, y_pred), 4)}$"
        f"\n$RMSE={round(pow(mean_squared_error(y_real, y_pred), 0.5), 4)}$"
        f"\n$MAPE={round(mean_absolute_percentage_error(y_real, y_pred), 4)}$"
        f"\n$R^2={round(r2_score(y_real, y_pred), 4)}$",
        verticalalignment="top",
        fontdict={"size": 16, "color": "k"},
    )  # 左上角显示模型性能指标
    ax.scatter(y_real, y_pred, c="none", marker="o", edgecolors="k")  # 散点图
    if fitting:
        from sklearn.linear_model import LinearRegression

        fitting_model = LinearRegression()
        fitting_model.fit([[item] for item in y_real], y_pred)
        ax.plot(
            [min(y_real), max(y_real)],
            [
                fitting_model.predict([[min(y_real)]]).item(),
                fitting_model.predict([[max(y_real)]]).item(),
            ],
            linewidth=2,
            linestyle="--",
            color="r",
            label="拟合曲线",
        )  # 拟合曲线
    ax.plot(
        [min(y_real), max(y_real)],
        [min(y_real), max(y_real)],
        linewidth=2,
        linestyle="-",
        color="r",
        label="参考曲线",
    )  # 参考曲线
    ax.set_title("真实值和预测值残差图", fontsize=24)  # 标题
    ax.set_xlabel("真实值", fontsize=20)  # x轴标签
    ax.set_ylabel("预测值", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    ax.legend(loc="lower right", prop={"size": 20})  # 图例

    plt.tight_layout()  # 防重叠
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示


def comparison_visualization(y_real, y_pred, output_path):
    # 绘制预测值与真实值的直观对比图
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)  # 定义画布

    ax.plot(y_real, marker=".", linestyle="-", linewidth=2, label="真实值")  # 画真实值
    ax.plot(y_pred, marker=".", linestyle="-", linewidth=2, label="预测值")  # 画预测值
    ax.set_title("真实值和预测值对比图", fontsize=24)  # 标题
    ax.set_xlabel("数据点", fontsize=20)  # x轴标签
    ax.set_ylabel("值", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    ax.legend(loc="best", prop={"size": 20})  # 图例

    plt.tight_layout()  # 防重叠
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示
