# 利用XGBoost和贝叶斯优化进行机器学习
# 修正版：可复现 + BO在训练集内部 + 修复 savemat None 报错
import os
import sys
import time
import random
import platform
import warnings

# ==================== 全局随机种子（必须在其他库导入前设置） ====================
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import sklearn
import skopt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from xgboost import XGBRegressor, plot_importance
from skopt import BayesSearchCV
from skopt.space import Integer, Real

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==================== 种子设置函数 ====================
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ==================== 全局配置（必须保持） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_X_PATH = os.path.join(BASE_DIR, "code_39", "res_39", "x_test39_1000.mat")
DATA_Y_PATH = os.path.join(BASE_DIR, "code_39", "res_39", "y_test39_1000.mat")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_bo_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 可复现：N_JOBS=1
N_JOBS = 1

# 数据划分
TEST_SIZE = 0.4
VAL_RATIO_IN_TEMP = 0.5

# 其他配置
BO_N_ITER = 20
BO_CV_FOLDS = 3
BO_SCORING = "neg_mean_squared_error"
EARLY_STOPPING_ROUNDS = 20
SEED = 42


# ==================== 数据加载 ====================
def load_mat_data(x_path, y_path):
    x_data = scipy.io.loadmat(x_path)["x_test"]
    y_data = scipy.io.loadmat(y_path)["y_test"]
    return x_data, y_data.flatten()


# ==================== 贝叶斯优化 ====================
def bayesian_optimization(x_train, y_train, seed=SEED):
    """
    使用贝叶斯优化确定最优超参数。
    注意：BO 在训练集内部做 CV，不使用验证集或测试集。
    """
    print("\n=== 贝叶斯超参数优化（在训练集内部做 CV） ===")

    search_spaces = {
        "max_depth": Integer(2, 6),
        "n_estimators": Integer(50, 500),
        "learning_rate": Real(0.03, 0.3, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "gamma": Real(0, 10),
        "min_child_weight": Integer(1, 30),
        "reg_alpha": Real(0, 5),
        "reg_lambda": Real(0, 30),
    }

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=N_JOBS,
        tree_method="hist",
    )

    cv_splitter = KFold(n_splits=BO_CV_FOLDS, shuffle=True, random_state=seed)

    bayes_cv = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=search_spaces,
        n_iter=BO_N_ITER,
        cv=cv_splitter,
        scoring=BO_SCORING,
        random_state=seed,
        verbose=0,
        n_jobs=N_JOBS,
    )

    opt_start = time.time()
    bayes_cv.fit(x_train, y_train)
    print(f"贝叶斯优化完成，耗时: {time.time() - opt_start:.2f}s")

    best_params = dict(bayes_cv.best_params_)
    print("\n找到的最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳CV分数 ({BO_SCORING}): {bayes_cv.best_score_:.4f}")
    print(f"best_index_ (在 cv_results_ 中的索引): {bayes_cv.best_index_}")

    return best_params


# ==================== 最终模型训练与评估 ====================
def train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, best_params):
    """
    用 BO 得到的最优超参数训练最终模型，并在验证集和测试集上评估。
    """
    params = dict(best_params)
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "rmse"

    # === 此处 final_model 写法必须保持 ===
    final_model = XGBRegressor(
        **params,
        n_jobs=N_JOBS,
        random_state=SEED,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # 早停放到 fit() 里（xgboost >= 2.0 推荐）
    fit_start = time.time()
    final_model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    training_time = time.time() - fit_start

    best_iteration = getattr(final_model, "best_iteration", None)
    best_score = getattr(final_model, "best_score", None)
    try:
        num_boosted_rounds = final_model.get_booster().num_boosted_rounds()
    except Exception:
        num_boosted_rounds = None

    print(f"\n训练完成，耗时: {training_time:.2f}s")
    print(f"best_iteration: {best_iteration}")
    print(f"best_score: {best_score}")
    print(f"实际训练轮数 (num_boosted_rounds): {num_boosted_rounds}")

    # ========== 验证集评估 ==========
    y_val_pred = final_model.predict(x_val)
    rmse_val = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    mae_val = float(mean_absolute_error(y_val, y_val_pred))
    mae_over_mean_val = float(mae_val / np.mean(y_val))
    r2_val = float(r2_score(y_val, y_val_pred))

    # ========== 测试集评估 ==========
    y_test_pred = final_model.predict(x_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mae_test = float(mean_absolute_error(y_test, y_test_pred))
    mae_over_mean_test = float(mae_test / np.mean(y_test))
    r2_test = float(r2_score(y_test, y_test_pred))

    # ========== 训练集评估 ==========
    y_train_pred = final_model.predict(x_train)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    mae_train = float(mean_absolute_error(y_train, y_train_pred))
    mae_over_mean_train = float(mae_train / np.mean(y_train))
    r2_train = float(r2_score(y_train, y_train_pred))

    print("\n验证集指标:")
    print(f"  MAE: {mae_val:.4f}")
    print(f"  RMSE: {rmse_val:.4f}")
    print(f"  MAE/Mean: {mae_over_mean_val:.4f}")
    print(f"  R²: {r2_val:.4f}")

    print("\n测试集指标:")
    print(f"  MAE: {mae_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  MAE/Mean: {mae_over_mean_test:.4f}")
    print(f"  R²: {r2_test:.4f}")

    print("\n训练集指标:")
    print(f"  MAE: {mae_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")
    print(f"  MAE/Mean: {mae_over_mean_train:.4f}")
    print(f"  R²: {r2_train:.4f}")

    rmse_gap = rmse_test - rmse_train
    rmse_gap_ratio = rmse_gap / rmse_train if rmse_train > 0 else np.nan
    print(f"\nRMSE 差距 (测试-训练): {rmse_gap:.4f} "
          f"({100 * rmse_gap_ratio:.2f}%)")

    return {
        "model": final_model,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "num_boosted_rounds": num_boosted_rounds,
        "training_time": training_time,
        # 验证集
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "mae_over_mean_val": mae_over_mean_val,
        "r2_val": r2_val,
        # 测试集
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "mae_over_mean_test": mae_over_mean_test,
        "r2_test": r2_test,
        # 训练集
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "mae_over_mean_train": mae_over_mean_train,
        "r2_train": r2_train,
        # 过拟合差距
        "rmse_gap": rmse_gap,
        "rmse_gap_ratio": rmse_gap_ratio,
        # 真实值和预测值（供 savemat 使用）
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,
    }


# ==================== 环境信息保存 ====================
def save_environment_info(best_params=None, results=None):
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "skopt": skopt.__version__,
        "N_JOBS": N_JOBS,
        "SEED": SEED,
        "TEST_SIZE": TEST_SIZE,
        "VAL_RATIO_IN_TEMP": VAL_RATIO_IN_TEMP,
        "BO_N_ITER": BO_N_ITER,
        "BO_CV_FOLDS": BO_CV_FOLDS,
        "BO_SCORING": BO_SCORING,
        "EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
        "best_params": best_params,
    }
    if results is not None:
        info["best_iteration"] = results["best_iteration"]
        info["best_score"] = results["best_score"]
        info["num_boosted_rounds"] = results["num_boosted_rounds"]
        info["rmse_val"] = results["rmse_val"]
        info["rmse_test"] = results["rmse_test"]
        info["rmse_train"] = results["rmse_train"]
        info["r2_val"] = results["r2_val"]
        info["r2_test"] = results["r2_test"]
        info["r2_train"] = results["r2_train"]

    path = os.path.join(OUTPUT_DIR, "environment_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print(f"\n✓ 环境信息已保存: {path}")


# ==================== 结果保存（已修复 None 报错） ====================
def save_results(results, best_params):
    # 1) 保存指标到 Excel
    metrics = {
        "指标": ["RMSE", "MAE", "MAE/Mean", "R²"],
        "训练集": [results["rmse_train"], results["mae_train"],
                 results["mae_over_mean_train"], results["r2_train"]],
        "验证集": [results["rmse_val"], results["mae_val"],
                 results["mae_over_mean_val"], results["r2_val"]],
        "测试集": [results["rmse_test"], results["mae_test"],
                 results["mae_over_mean_test"], results["r2_test"]],
    }
    metrics_df = pd.DataFrame(metrics)

    excel_path = os.path.join(OUTPUT_DIR, "final_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        metrics_df.to_excel(writer, sheet_name="指标汇总", index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name="最优超参数", index=False)
    print(f"✓ Excel 已保存: {excel_path}")

    # 2) 保存 .mat（全部为数值，不再有 None）
    mat_path = os.path.join(OUTPUT_DIR, "res_XG.mat")
    scipy.io.savemat(mat_path, {
        "residuals_test": results["y_test"] - results["y_test_pred"],
        "residuals_val": results["y_val"] - results["y_val_pred"],
        "residuals_train": results["y_train"] - results["y_train_pred"],
        "y_test": results["y_test"],
        "y_val": results["y_val"],
        "y_train": results["y_train"],
        "y_test_pred": results["y_test_pred"],
        "y_val_pred": results["y_val_pred"],
        "y_train_pred": results["y_train_pred"],
    })
    print(f"✓ MAT 已保存: {mat_path}")


# ==================== 可视化 ====================
def plot_training_curve(model, output_dir):
    """训练过程曲线"""
    try:
        results = model.evals_result()
        eval_keys = list(results.keys())
        if len(eval_keys) < 1:
            print("⚠ 没有 eval 结果，跳过训练曲线")
            return

        val_key = eval_keys[0]
        train_key = eval_keys[1] if len(eval_keys) > 1 else None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results[val_key]["rmse"], label="Validation", color="red")
        if train_key is not None:
            ax.plot(results[train_key]["rmse"], label="Train", color="green")

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            ax.axvline(best_iter, color="gray", linestyle="--",
                       label=f"best_iteration = {best_iter}")

        ax.set_xlabel("Boosting Rounds")
        ax.set_ylabel("RMSE")
        ax.set_title("训练过程 (RMSE)")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        save_path = os.path.join(output_dir, "training_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ 训练曲线已保存: {save_path}")
    except Exception as e:
        print(f"⚠ 训练曲线绘制失败: {e}")


def plot_prediction_scatter(y_true, y_pred, r2, output_dir, name="test"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"预测 vs 真实 ({name})")
    ax.text(0.05, 0.9, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"prediction_scatter_{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 预测散点图已保存: {save_path}")


def plot_residuals(y_true, y_pred, output_dir, name="test"):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"残差分布 ({name})")
    axes[0].set_xlabel("Residuals")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.4)

    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_title(f"残差 vs 预测值 ({name})")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"residuals_{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 残差图已保存: {save_path}")


# ==================== 主程序 ====================
def main():
    set_global_seed(SEED)

    print("=" * 70)
    print("XGBoost + 贝叶斯优化：完整流程")
    print("=" * 70)

    print("\n加载数据...")
    x, y = load_mat_data(DATA_X_PATH, DATA_Y_PATH)
    print(f"数据加载完成: x={x.shape}, y={y.shape}")

    # 数据划分
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=VAL_RATIO_IN_TEMP, random_state=SEED
    )
    print(f"Data sizes - Train: {x_train.shape[0]}, "
          f"Valid: {x_val.shape[0]}, Test: {x_test.shape[0]}")
    # 特征选择 
    k_features = max(3, int(x_train.shape[1] * 0.6))  # 大数据集保留50%特征
    selector = SelectKBest(mutual_info_regression, k=k_features)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_val_selected = selector.transform(x_val)
    x_test_selected = selector.transform(x_test)

    # ========== 贝叶斯优化（在训练集内部做 CV） ==========
    best_params = bayesian_optimization(x_train_selected, y_train, seed=SEED)

    # ========== 最终模型训练与评估 ==========
    results = train_and_evaluate(
        x_train_selected, y_train, x_val_selected, y_val, x_test_selected, y_test, best_params
    )

    # ========== 保存环境信息和结果 ==========
    save_environment_info(best_params, results)
    save_results(results, best_params)

    # ========== 可视化 ==========
    print("\n生成可视化图表...")
    plot_training_curve(results["model"], OUTPUT_DIR)
    plot_prediction_scatter(y_test, results["y_test_pred"], results["r2_test"],
                           OUTPUT_DIR, name="test")
    plot_prediction_scatter(y_val, results["y_val_pred"], results["r2_val"],
                           OUTPUT_DIR, name="val")
    plot_residuals(y_test, results["y_test_pred"], OUTPUT_DIR, name="test")
    plot_residuals(y_val, results["y_val_pred"], OUTPUT_DIR, name="val")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    


if __name__ == "__main__":
    main()