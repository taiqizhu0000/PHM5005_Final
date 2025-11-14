#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段2: 嵌套交叉验证模型训练
- 外层CV: 5-Fold StratifiedKFold (重复3次)
- 内层CV: 5-Fold StratifiedKFold (超参数调优)
- Pipeline: StandardScaler + SelectKBest + ElasticNet LogisticRegression  
- 评估指标: AUROC, AUPRC, F1, Recall, Specificity
- 记录特征选择和系数用于后续分析
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, recall_score, precision_score,
                             accuracy_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 配置路径
BASE_DIR = "/Users/a/Desktop/5005"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "stage2_nested_cv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("阶段2: 嵌套交叉验证模型训练")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1] 加载数据...")

X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'), allow_pickle=True)
X_test = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'), allow_pickle=True)
y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'), allow_pickle=True)
y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'), allow_pickle=True)

with open(os.path.join(RESULTS_DIR, 'feature_names.json'), 'r') as f:
    feature_info = json.load(f)
    feature_names = feature_info['all_features']

print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")
print(f"  特征数: {len(feature_names)}")

# ============================================================================
# 2. 定义评估函数
# ============================================================================
def calculate_metrics(y_true, y_pred, y_proba):
    """计算所有评估指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_proba),
        'auprc': average_precision_score(y_true, y_proba),
        'f1': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),  # Sensitivity
        'precision': precision_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'accuracy': accuracy_score(y_true, y_pred),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }
    return metrics

# ============================================================================
# 3. 定义Pipeline和超参数网格
# ============================================================================
print("\n[2] 定义Pipeline和超参数网格...")

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif)),
    ('classifier', LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        class_weight='balanced',
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

# 超参数网格
param_grid = {
    'selector__k': [300, 500, 700, 'all'],  # None改为'all'字符串
    'classifier__l1_ratio': [0.2, 0.5, 0.8],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

print(f"  Pipeline: {[name for name, _ in pipeline.steps]}")
print(f"  超参数组合数: {len(param_grid['selector__k']) * len(param_grid['classifier__l1_ratio']) * len(param_grid['classifier__C'])}")

# ============================================================================
# 4. 嵌套交叉验证
# ============================================================================
print("\n[3] 执行嵌套交叉验证...")
print(f"  外层: 5-Fold × 3 重复 = 15次训练")
print(f"  内层: 5-Fold GridSearchCV")

# 外层交叉验证 (重复3次)
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

# 内层交叉验证
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 存储结果
outer_results = []
feature_selection_history = []  # 记录每次选择的特征
coefficient_history = []  # 记录每次的系数
best_params_history = []  # 记录每次最佳参数

fold_idx = 0
print("\n  开始训练...")

for train_idx, val_idx in tqdm(outer_cv.split(X_train, y_train), total=15, desc="外层CV"):
    fold_idx += 1
    
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # 内层GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring='roc_auc',  # 使用AUROC作为优化目标
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_fold, y_train_fold)
    
    # 最佳模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_params_history.append({
        'fold': fold_idx,
        **best_params
    })
    
    # 预测验证集
    y_val_pred = best_model.predict(X_val_fold)
    y_val_proba = best_model.predict_proba(X_val_fold)[:, 1]
    
    # 计算指标
    metrics = calculate_metrics(y_val_fold, y_val_pred, y_val_proba)
    metrics['fold'] = fold_idx
    metrics['best_params'] = best_params
    outer_results.append(metrics)
    
    # 记录特征选择
    selector = best_model.named_steps['selector']
    if hasattr(selector, 'get_support'):
        selected_features_mask = selector.get_support()
        selected_features = [feature_names[i] for i, selected in enumerate(selected_features_mask) if selected]
    else:
        selected_features = feature_names  # 如果k='all'
    
    feature_selection_history.append({
        'fold': fold_idx,
        'n_features': len(selected_features),
        'features': selected_features
    })
    
    # 记录系数
    classifier = best_model.named_steps['classifier']
    coefficients = classifier.coef_[0]
    
    # 如果使用了特征选择，需要映射回原始特征
    if hasattr(selector, 'get_support'):
        full_coef = np.zeros(len(feature_names))
        full_coef[selected_features_mask] = coefficients
    else:
        full_coef = coefficients
    
    coefficient_history.append({
        'fold': fold_idx,
        'coefficients': full_coef.tolist(),
        'non_zero_count': int(np.sum(full_coef != 0))
    })

print(f"\n  完成15次外层交叉验证训练")

# ============================================================================
# 5. 汇总外层CV结果
# ============================================================================
print("\n[4] 汇总外层交叉验证结果...")

results_df = pd.DataFrame(outer_results)

# 计算平均指标
mean_metrics = {
    'auroc_mean': results_df['auroc'].mean(),
    'auroc_std': results_df['auroc'].std(),
    'auprc_mean': results_df['auprc'].mean(),
    'auprc_std': results_df['auprc'].std(),
    'f1_mean': results_df['f1'].mean(),
    'f1_std': results_df['f1'].std(),
    'recall_mean': results_df['recall'].mean(),
    'recall_std': results_df['recall'].std(),
    'specificity_mean': results_df['specificity'].mean(),
    'specificity_std': results_df['specificity'].std(),
    'accuracy_mean': results_df['accuracy'].mean(),
    'accuracy_std': results_df['accuracy'].std(),
}

print(f"\n  外层CV性能 (Mean ± Std):")
print(f"    AUROC:       {mean_metrics['auroc_mean']:.4f} ± {mean_metrics['auroc_std']:.4f}")
print(f"    AUPRC:       {mean_metrics['auprc_mean']:.4f} ± {mean_metrics['auprc_std']:.4f}")
print(f"    F1-Score:    {mean_metrics['f1_mean']:.4f} ± {mean_metrics['f1_std']:.4f}")
print(f"    Recall:      {mean_metrics['recall_mean']:.4f} ± {mean_metrics['recall_std']:.4f}")
print(f"    Specificity: {mean_metrics['specificity_mean']:.4f} ± {mean_metrics['specificity_std']:.4f}")
print(f"    Accuracy:    {mean_metrics['accuracy_mean']:.4f} ± {mean_metrics['accuracy_std']:.4f}")

# ============================================================================
# 6. 在测试集上评估
# ============================================================================
print("\n[5] 在测试集上评估最终模型...")

# 使用全部训练数据训练最终模型
# 使用最常出现的最佳参数
best_params_df = pd.DataFrame(best_params_history)
mode_params = {}
for col in ['selector__k', 'classifier__l1_ratio', 'classifier__C']:
    mode_params[col] = best_params_df[col].mode()[0]

print(f"  使用最常见的最佳参数: {mode_params}")

# 训练最终模型
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=mode_params['selector__k'])),
    ('classifier', LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        l1_ratio=mode_params['classifier__l1_ratio'],
        C=mode_params['classifier__C'],
        class_weight='balanced',
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

final_pipeline.fit(X_train, y_train)

# 测试集预测
y_test_pred = final_pipeline.predict(X_test)
y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

# 计算测试集指标
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

print(f"\n  测试集性能:")
print(f"    AUROC:       {test_metrics['auroc']:.4f}")
print(f"    AUPRC:       {test_metrics['auprc']:.4f}")
print(f"    F1-Score:    {test_metrics['f1']:.4f}")
print(f"    Recall:      {test_metrics['recall']:.4f}")
print(f"    Specificity: {test_metrics['specificity']:.4f}")
print(f"    Accuracy:    {test_metrics['accuracy']:.4f}")

# ============================================================================
# 7. 保存结果
# ============================================================================
print("\n[6] 保存结果...")

# 保存外层CV结果
results_df.to_csv(os.path.join(OUTPUT_DIR, 'outer_cv_results.csv'), index=False)

# 保存特征选择历史
with open(os.path.join(OUTPUT_DIR, 'feature_selection_history.json'), 'w') as f:
    json.dump(feature_selection_history, f, indent=2)

# 保存系数历史
with open(os.path.join(OUTPUT_DIR, 'coefficient_history.json'), 'w') as f:
    json.dump(coefficient_history, f, indent=2)

# 保存最佳参数历史
best_params_df.to_csv(os.path.join(OUTPUT_DIR, 'best_params_history.csv'), index=False)

# 保存测试集结果
with open(os.path.join(OUTPUT_DIR, 'test_set_results.json'), 'w') as f:
    json.dump(test_metrics, f, indent=2)

# 保存最终模型
with open(os.path.join(OUTPUT_DIR, 'final_model.pkl'), 'wb') as f:
    pickle.dump(final_pipeline, f)

# 保存测试集预测
test_predictions = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_test_pred,
    'y_proba': y_test_proba
})
test_predictions.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

# 保存汇总指标
summary = {
    'outer_cv_metrics': mean_metrics,
    'test_set_metrics': test_metrics,
    'final_model_params': mode_params,
    'n_outer_folds': 15,
    'n_inner_folds': 5
}

with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  结果已保存到: {OUTPUT_DIR}")

# ============================================================================
# 8. 可视化
# ============================================================================
print("\n[7] 生成可视化...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8.1 外层CV性能分布（箱线图）
ax1 = fig.add_subplot(gs[0, :2])
metrics_to_plot = ['auroc', 'auprc', 'f1', 'recall', 'specificity']
data_to_plot = [results_df[m].values for m in metrics_to_plot]
bp = ax1.boxplot(data_to_plot, labels=['AUROC', 'AUPRC', 'F1', 'Recall', 'Specificity'],
                 patch_artist=True, showmeans=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Outer CV Performance Distribution (15 folds)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.05])

# 8.2 超参数分布
ax2 = fig.add_subplot(gs[0, 2])
param_counts = best_params_df['classifier__l1_ratio'].value_counts().sort_index()
ax2.bar(param_counts.index.astype(str), param_counts.values, color='#e74c3c', alpha=0.7)
ax2.set_xlabel('l1_ratio', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Best l1_ratio Distribution', fontsize=12, fontweight='bold')

# 8.3 测试集ROC曲线
ax3 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax3.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC (AUC={test_metrics["auroc"]:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax3.set_xlabel('False Positive Rate', fontsize=11)
ax3.set_ylabel('True Positive Rate', fontsize=11)
ax3.set_title('Test Set ROC Curve', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# 8.4 测试集PR曲线
ax4 = fig.add_subplot(gs[1, 1])
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
ax4.plot(recall_vals, precision_vals, color='#9b59b6', lw=2, label=f'PR (AUC={test_metrics["auprc"]:.3f})')
baseline = (y_test == 1).sum() / len(y_test)
ax4.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
ax4.set_xlabel('Recall', fontsize=11)
ax4.set_ylabel('Precision', fontsize=11)
ax4.set_title('Test Set Precision-Recall Curve', fontsize=12, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(alpha=0.3)

# 8.5 测试集混淆矩阵
ax5 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax5,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
ax5.set_xlabel('Predicted', fontsize=11)
ax5.set_ylabel('Actual', fontsize=11)
ax5.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold')

# 8.6 特征选择数量分布
ax6 = fig.add_subplot(gs[2, 0])
n_features_selected = [fs['n_features'] for fs in feature_selection_history]
ax6.hist(n_features_selected, bins=15, color='#1abc9c', alpha=0.7, edgecolor='black')
ax6.axvline(np.mean(n_features_selected), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(n_features_selected):.0f}')
ax6.set_xlabel('Number of Features Selected', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Feature Selection Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 8.7 C参数分布（对数刻度）
ax7 = fig.add_subplot(gs[2, 1])
c_values = best_params_df['classifier__C'].value_counts().sort_index()
ax7.bar(range(len(c_values)), c_values.values, color='#f39c12', alpha=0.7)
ax7.set_xticks(range(len(c_values)))
ax7.set_xticklabels([f'{c:.3f}' for c in c_values.index], rotation=45)
ax7.set_xlabel('C (Regularization)', fontsize=11)
ax7.set_ylabel('Frequency', fontsize=11)
ax7.set_title('Best C Distribution', fontsize=12, fontweight='bold')

# 8.8 非零系数数量
ax8 = fig.add_subplot(gs[2, 2])
non_zero_counts = [ch['non_zero_count'] for ch in coefficient_history]
ax8.hist(non_zero_counts, bins=15, color='#e67e22', alpha=0.7, edgecolor='black')
ax8.axvline(np.mean(non_zero_counts), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(non_zero_counts):.0f}')
ax8.set_xlabel('Number of Non-zero Coefficients', fontsize=11)
ax8.set_ylabel('Frequency', fontsize=11)
ax8.set_title('Sparsity Distribution', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(axis='y', alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, 'nested_cv_results.png'), dpi=300, bbox_inches='tight')
print(f"  可视化已保存: nested_cv_results.png")
plt.close()

# ============================================================================
# 9. 生成报告
# ============================================================================
print("\n[8] 生成报告...")

report = f"""# 阶段2: 嵌套交叉验证报告

## 模型配置

- **Pipeline**: StandardScaler → SelectKBest → ElasticNet Logistic Regression
- **外层CV**: 5-Fold × 3 重复 = 15次训练
- **内层CV**: 5-Fold GridSearchCV
- **超参数空间**:
  - selector__k: [300, 500, 700, all]
  - classifier__l1_ratio: [0.2, 0.5, 0.8]
  - classifier__C: [0.001, 0.01, 0.1, 1, 10, 100]
  - 总组合数: {len(param_grid['selector__k']) * len(param_grid['classifier__l1_ratio']) * len(param_grid['classifier__C'])}

## 外层交叉验证性能

### 主要指标 (Mean ± Std)

| 指标 | 均值 | 标准差 |
|------|------|--------|
| **AUROC** | {mean_metrics['auroc_mean']:.4f} | {mean_metrics['auroc_std']:.4f} |
| **AUPRC** | {mean_metrics['auprc_mean']:.4f} | {mean_metrics['auprc_std']:.4f} |
| **F1-Score** | {mean_metrics['f1_mean']:.4f} | {mean_metrics['f1_std']:.4f} |
| **Recall (Sensitivity)** | {mean_metrics['recall_mean']:.4f} | {mean_metrics['recall_std']:.4f} |
| **Specificity** | {mean_metrics['specificity_mean']:.4f} | {mean_metrics['specificity_std']:.4f} |
| **Accuracy** | {mean_metrics['accuracy_mean']:.4f} | {mean_metrics['accuracy_std']:.4f} |

### 解读

- **AUROC > 0.7**: 模型具有良好的区分能力
- **AUPRC**: 考虑类别不平衡，PR曲线下面积
- **Recall**: 高风险患者的召回率（灵敏度）
- **Specificity**: 低风险患者的正确识别率

## 测试集性能

| 指标 | 值 |
|------|------|
| **AUROC** | {test_metrics['auroc']:.4f} |
| **AUPRC** | {test_metrics['auprc']:.4f} |
| **F1-Score** | {test_metrics['f1']:.4f} |
| **Recall** | {test_metrics['recall']:.4f} |
| **Precision** | {test_metrics['precision']:.4f} |
| **Specificity** | {test_metrics['specificity']:.4f} |
| **Accuracy** | {test_metrics['accuracy']:.4f} |

### 混淆矩阵

|  | 预测: 低风险 | 预测: 高风险 |
|---|---|---|
| **实际: 低风险** | {test_metrics['tn']} (TN) | {test_metrics['fp']} (FP) |
| **实际: 高风险** | {test_metrics['fn']} (FN) | {test_metrics['tp']} (TP) |

## 最优超参数

最常见的最佳参数组合:
- **selector__k**: {mode_params['selector__k']}
- **classifier__l1_ratio**: {mode_params['classifier__l1_ratio']}
- **classifier__C**: {mode_params['classifier__C']}

## 特征选择统计

- **平均选择特征数**: {np.mean(n_features_selected):.0f} ± {np.std(n_features_selected):.0f}
- **平均非零系数数**: {np.mean(non_zero_counts):.0f} ± {np.std(non_zero_counts):.0f}
- **稀疏化比例**: {(1 - np.mean(non_zero_counts) / len(feature_names)) * 100:.1f}%

## 输出文件

- `outer_cv_results.csv`: 15次外层CV的详细结果
- `feature_selection_history.json`: 每次选择的特征列表
- `coefficient_history.json`: 每次的模型系数
- `best_params_history.csv`: 每次的最佳超参数
- `test_set_results.json`: 测试集评估结果
- `test_predictions.csv`: 测试集预测值
- `final_model.pkl`: 最终训练的模型
- `summary.json`: 汇总统计
- `nested_cv_results.png`: 可视化结果

---
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(os.path.join(OUTPUT_DIR, 'stage2_report.md'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  报告已保存: stage2_report.md")

print("\n" + "=" * 80)
print("阶段2 完成！")
print("=" * 80)
print(f"\n输出目录: {OUTPUT_DIR}")
print("\n下一步: 运行 stage3_feature_importance.py 进行特征重要性分析")

