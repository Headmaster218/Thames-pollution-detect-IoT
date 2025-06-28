
# AquaSense · Machine-Learning Module  
*Lightweight AI for Field-Deployable E. coli Estimation*

> **Core idea:** 以 **五项低成本物理化学指标**（O₂、温度、pH、浊度、导电率）为特征，用 18 k 参数的小型神经网络在现场近实时估计 **E. coli** 含量，免去 18-24 h 培养等待 —— 为快速水质预警提供可落地方案。

---

## 1 · 任务定义
- **输入**：44 维 proxy 特征（原始 5 项＋统计 / trigonometric 派生特征），已做 log 或标准化转换  
- **输出**：`ln Coliform` —— 对 E. coli CFU · 100 mL⁻¹ 的自然对数  

---

## 2 · 模型结构

| 层级 | 输出维度 | 细节 |
|------|----------|------|
| FC-1 | 128 | LeakyReLU(α=0.01) + Dropout(0.25) |
| FC-2 | 64 | LeakyReLU + Dropout(0.25) |
| FC-3 | 32 | LeakyReLU + Dropout(0.25) |
| Output | 1 | 线性激活 |

*优化器* Adam (lr 1e-4, weight_decay 1e-4) | *损失函数* HuberLoss  
*LR 调度* StepLR(step = 100, γ = 0.9) | *早停* patience = 50

> **参数总量** ≈ 18 k —— 适合 Raspberry Pi / ESP-32 边缘设备部署。


---

## 3 · 性能对比

| 方法                       | RMSE ↓   | MAE ↓    | R² ↑     |
| ------------------------ | -------- | -------- | -------- |
| Turbidity-only LR        | 1.98     | 1.62     | 0.12     |
| Linear Regression        | 1.95     | 1.54     | 0.14     |
| Random Forest            | 1.74     | 1.34     | 0.32     |
| XGBoost                  | 1.79     | 1.37     | 0.28     |
| MLP-2L                   | 1.88     | 1.41     | 0.20     |
| **CNN-DropHuber (Ours)** | **0.82** | **0.72** | **0.37** |

<details>
<summary>点击查看 Feature Importance (XGBoost)</summary>

![feat](3.3xgboost_importance_result.png)

</details>

---

## 4 · 输出文件

| 路径                                | 描述                         |
| --------------------------------- | -------------------------- |
| `model_best_val_loss.pth` | 最佳权重 (val RMSE 最低)         |
| `scaler.pkl`                      | StandardScaler 对象 (训练阶段保存) |
| `baseline_results.csv`            | 所有对比模型 RMSE / MAE / R²     |
| `3.1loss+R2_whole.png`              | 训练-验证损失曲线                  |
| `pred_true_scatter.png`           | 预测 vs 真值散点图                |


---

## 5 · 联系

项目负责人：**John Wu**

Issue / PR 欢迎直接在 GitHub 提交。
