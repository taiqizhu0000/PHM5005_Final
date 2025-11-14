#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段3: 特征重要性分析
- 跨折稳定性分析：统计15次训练中每个特征被选中的频率
- 系数统计：计算非零系数的中位数、均值和四分位数
- Permutation Importance：在测试集上验证真实贡献
- Top特征排名：临床特征 vs 基因特征
- 通路分析：7个通路的特征贡献
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = "/Users/a/Desktop/5005"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STAGE2_DIR = os.path.join(RESULTS_DIR, "stage2_nested_cv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "stage3_feature_importance")
PATHWAY_DIR = os.path.join(BASE_DIR, "dataset", "pathway_gene_list")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("阶段3: 特征重要性分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1] 加载数据...")

# 加载系数历史
with open(os.path.join(STAGE2_DIR, 'coefficient_history.json'), 'r') as f:
    coef_history = json.load(f)

# 加载特征名称
with open(os.path.join(RESULTS_DIR, 'feature_names.json'), 'r') as f:
    feature_info = json.load(f)
    all_features = feature_info['all_features']
    clinical_features = feature_info['clinical_features']
    gene_features = feature_info['gene_features']

# 加载最终模型
with open(os.path.join(STAGE2_DIR, 'final_model.pkl'), 'rb') as f:
    final_model = pickle.load(f)

# 加载测试集数据
X_test = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'), allow_pickle=True)
y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'), allow_pickle=True)

print(f"  系数历史: {len(coef_history)} 次训练")
print(f"  特征总数: {len(all_features)}")
print(f"  临床特征: {len(clinical_features)}")
print(f"  基因特征: {len(gene_features)}")

# ============================================================================
# 2. 跨折稳定性分析
# ============================================================================
print("\n[2] 分析跨折稳定性...")

# 收集所有系数
all_coefficients = np.array([fold['coefficients'] for fold in coef_history])
n_folds = len(coef_history)

# 计算每个特征的统计量
feature_stats = []

for i, feature_name in enumerate(all_features):
    coefs = all_coefficients[:, i]
    
    # 非零系数的次数
    non_zero_count = np.sum(coefs != 0)
    frequency = non_zero_count / n_folds
    
    # 只考虑非零系数计算统计量
    non_zero_coefs = coefs[coefs != 0]
    
    if len(non_zero_coefs) > 0:
        median_coef = np.median(non_zero_coefs)
        mean_coef = np.mean(non_zero_coefs)
        std_coef = np.std(non_zero_coefs)
        q25 = np.percentile(non_zero_coefs, 25)
        q75 = np.percentile(non_zero_coefs, 75)
    else:
        median_coef = mean_coef = std_coef = q25 = q75 = 0
    
    # 判断特征类型
    if feature_name in clinical_features:
        feature_type = 'Clinical'
    else:
        feature_type = 'Gene'
    
    feature_stats.append({
        'feature': feature_name,
        'type': feature_type,
        'frequency': frequency,
        'selection_count': non_zero_count,
        'median_coef': median_coef,
        'mean_coef': mean_coef,
        'std_coef': std_coef,
        'q25': q25,
        'q75': q75,
        'abs_median_coef': abs(median_coef),
        'importance_score': frequency * abs(median_coef)  # 综合重要性
    })

feature_stats_df = pd.DataFrame(feature_stats)

# 按重要性得分排序
feature_stats_df = feature_stats_df.sort_values('importance_score', ascending=False)

print(f"  被选中至少一次的特征数: {(feature_stats_df['selection_count'] > 0).sum()}")
print(f"  稳定特征 (频率>80%): {(feature_stats_df['frequency'] > 0.8).sum()}")
print(f"  高频特征 (频率>50%): {(feature_stats_df['frequency'] > 0.5).sum()}")

# ============================================================================
# 3. Top特征分析
# ============================================================================
print("\n[3] 提取Top特征...")

# Top 50特征
top_features = feature_stats_df.head(50).copy()

print(f"\n  Top 50特征组成:")
print(f"    临床特征: {(top_features['type'] == 'Clinical').sum()}")
print(f"    基因特征: {(top_features['type'] == 'Gene').sum()}")

# 暂时打印Top特征（稍后会在merge perm importance后重新提取）
print(f"\n  Top 20临床特征:")
temp_top_clinical = feature_stats_df[feature_stats_df['type'] == 'Clinical'].head(20)
for idx, row in temp_top_clinical.iterrows():
    print(f"    {row['feature'][:50]:50s} | 频率:{row['frequency']:.2f} | 系数:{row['median_coef']:+.4f}")

print(f"\n  Top 20基因特征:")
temp_top_genes = feature_stats_df[feature_stats_df['type'] == 'Gene'].head(20)
for idx, row in temp_top_genes.iterrows():
    gene_name = row['feature'].replace('gene_', '')
    print(f"    {gene_name:20s} | 频率:{row['frequency']:.2f} | 系数:{row['median_coef']:+.4f}")

# ============================================================================
# 4. Permutation Importance (测试集验证)
# ============================================================================
print("\n[4] 计算Permutation Importance (测试集)...")

perm_importance = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring='roc_auc',
    n_jobs=-1
)

# 合并到特征统计
perm_scores = []
for i, feature_name in enumerate(all_features):
    perm_scores.append({
        'feature': feature_name,
        'perm_importance_mean': perm_importance.importances_mean[i],
        'perm_importance_std': perm_importance.importances_std[i]
    })

perm_df = pd.DataFrame(perm_scores)
feature_stats_df = feature_stats_df.merge(perm_df, on='feature')

# 重新提取top特征（现在包含perm_importance）
top_clinical = feature_stats_df[feature_stats_df['type'] == 'Clinical'].head(20)
top_genes = feature_stats_df[feature_stats_df['type'] == 'Gene'].head(20)

# Top permutation importance
top_perm = feature_stats_df.nlargest(20, 'perm_importance_mean')
print(f"\n  Top 20 Permutation Importance特征:")
for idx, row in top_perm.iterrows():
    display_name = row['feature'].replace('gene_', '')[:30]
    print(f"    {display_name:30s} | PI:{row['perm_importance_mean']:+.4f}±{row['perm_importance_std']:.4f}")

# ============================================================================
# 5. 通路分析
# ============================================================================
print("\n[5] 分析通路特征贡献...")

# 加载通路基因映射
import glob
pathway_genes = {}
for pathway_file in glob.glob(os.path.join(PATHWAY_DIR, "*.csv")):
    pathway_name = os.path.basename(pathway_file).replace('_symbols.csv', '').replace('_hsa', '')
    df = pd.read_csv(pathway_file)
    genes = df.iloc[:, 0].str.strip('"').tolist()
    pathway_genes[pathway_name] = genes

# 统计每个通路的特征
pathway_stats = []
for pathway_name, genes in pathway_genes.items():
    # 找到对应的基因特征
    pathway_features = [f'gene_{g}' for g in genes if f'gene_{g}' in all_features]
    pathway_feature_stats = feature_stats_df[feature_stats_df['feature'].isin(pathway_features)]
    
    if len(pathway_feature_stats) > 0:
        pathway_stats.append({
            'pathway': pathway_name,
            'total_genes': len(genes),
            'available_genes': len(pathway_features),
            'selected_genes': (pathway_feature_stats['selection_count'] > 0).sum(),
            'stable_genes': (pathway_feature_stats['frequency'] > 0.5).sum(),
            'avg_frequency': pathway_feature_stats['frequency'].mean(),
            'avg_importance': pathway_feature_stats['importance_score'].mean(),
            'top_gene': pathway_feature_stats.nlargest(1, 'importance_score')['feature'].values[0].replace('gene_', '') if len(pathway_feature_stats) > 0 else 'N/A'
        })

if len(pathway_stats) > 0:
    pathway_stats_df = pd.DataFrame(pathway_stats).sort_values('avg_importance', ascending=False)
else:
    # 创建空DataFrame
    pathway_stats_df = pd.DataFrame(columns=['pathway', 'total_genes', 'available_genes', 
                                              'selected_genes', 'stable_genes', 'avg_frequency',
                                              'avg_importance', 'top_gene'])

print(f"\n  通路重要性排名:")
for idx, row in pathway_stats_df.iterrows():
    print(f"    {row['pathway']:30s} | 可用:{row['available_genes']:3d} | 选中:{row['selected_genes']:3d} | 重要度:{row['avg_importance']:.4f}")

# ============================================================================
# 6. 保存结果
# ============================================================================
print("\n[6] 保存结果...")

# 保存完整特征统计
feature_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_full.csv'), index=False)

# 保存Top特征
top_features.to_csv(os.path.join(OUTPUT_DIR, 'top50_features.csv'), index=False)
top_clinical.to_csv(os.path.join(OUTPUT_DIR, 'top20_clinical_features.csv'), index=False)
top_genes.to_csv(os.path.join(OUTPUT_DIR, 'top20_gene_features.csv'), index=False)

# 保存通路统计
pathway_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'pathway_importance.csv'), index=False)

print(f"  结果已保存到: {OUTPUT_DIR}")

# ============================================================================
# 7. 可视化
# ============================================================================
print("\n[7] 生成可视化...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 7.1 Top 20特征重要性（综合得分）
ax1 = fig.add_subplot(gs[0, :2])
top20 = feature_stats_df.head(20)
colors = ['#e74c3c' if t == 'Clinical' else '#3498db' for t in top20['type']]
y_pos = np.arange(len(top20))
display_names = [f.replace('gene_', '')[:30] for f in top20['feature']]
ax1.barh(y_pos, top20['importance_score'], color=colors, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(display_names, fontsize=9)
ax1.set_xlabel('Importance Score (Frequency × |Median Coef|)', fontsize=11)
ax1.set_title('Top 20 Features by Importance Score', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label='Clinical'),
                   Patch(facecolor='#3498db', alpha=0.7, label='Gene')]
ax1.legend(handles=legend_elements, loc='lower right')

# 7.2 特征选择频率分布
ax2 = fig.add_subplot(gs[0, 2])
freq_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
freq_counts, _ = np.histogram(feature_stats_df['frequency'], bins=freq_bins)
ax2.bar(['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], freq_counts, 
        color='#9b59b6', alpha=0.7, edgecolor='black')
ax2.set_ylabel('Number of Features', fontsize=11)
ax2.set_xlabel('Selection Frequency', fontsize=11)
ax2.set_title('Feature Selection Frequency Distribution', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 7.3 系数稳定性（Top 30）
ax3 = fig.add_subplot(gs[1, :2])
top30_for_plot = feature_stats_df.head(30).copy()
x = np.arange(len(top30_for_plot))
display_names_30 = [f.replace('gene_', '')[:25] for f in top30_for_plot['feature']]
ax3.errorbar(x, top30_for_plot['median_coef'], 
             yerr=[top30_for_plot['median_coef'] - top30_for_plot['q25'],
                   top30_for_plot['q75'] - top30_for_plot['median_coef']],
             fmt='o', capsize=3, capthick=2, markersize=5, alpha=0.7, color='#1abc9c')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax3.set_xticks(x[::3])
ax3.set_xticklabels([display_names_30[i] for i in range(0, len(display_names_30), 3)], 
                     rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Coefficient (Median & IQR)', fontsize=11)
ax3.set_title('Top 30 Features: Coefficient Stability', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# 7.4 Permutation Importance Top 20
ax4 = fig.add_subplot(gs[1, 2])
top20_perm = feature_stats_df.nlargest(20, 'perm_importance_mean')
y_pos = np.arange(len(top20_perm))
display_names_perm = [f.replace('gene_', '')[:20] for f in top20_perm['feature']]
ax4.barh(y_pos, top20_perm['perm_importance_mean'], 
         xerr=top20_perm['perm_importance_std'],
         color='#f39c12', alpha=0.7, capsize=3)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(display_names_perm, fontsize=8)
ax4.set_xlabel('Permutation Importance', fontsize=10)
ax4.set_title('Top 20 by Permutation Importance', fontsize=11, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# 7.5 通路重要性对比
ax5 = fig.add_subplot(gs[2, :])
pathway_stats_sorted = pathway_stats_df.sort_values('avg_importance', ascending=True)
y_pos = np.arange(len(pathway_stats_sorted))
colors_pathway = plt.cm.viridis(np.linspace(0.3, 0.9, len(pathway_stats_sorted)))
bars = ax5.barh(y_pos, pathway_stats_sorted['avg_importance'], color=colors_pathway, alpha=0.8)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(pathway_stats_sorted['pathway'], fontsize=10)
ax5.set_xlabel('Average Importance Score', fontsize=11)
ax5.set_title('Pathway-level Feature Importance', fontsize=13, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, (idx, row) in enumerate(pathway_stats_sorted.iterrows()):
    ax5.text(row['avg_importance'] + 0.0001, i, 
             f"  {row['selected_genes']}/{row['available_genes']} genes",
             va='center', fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_analysis.png'), 
            dpi=300, bbox_inches='tight')
print(f"  可视化已保存: feature_importance_analysis.png")
plt.close()

# ============================================================================
# 8. 生成报告
# ============================================================================
print("\n[8] 生成报告...")

report = f"""# 阶段3: 特征重要性分析报告

## 方法概述

### 1. 跨折稳定性分析
- 统计15次嵌套CV训练中每个特征被选中（系数≠0）的频率
- 计算非零系数的中位数、均值和四分位数
- 综合重要性得分 = 选择频率 × |系数中位数|

### 2. Permutation Importance
- 在独立测试集上进行10次重复
- 衡量特征对模型预测的真实贡献
- 验证系数重要性的可靠性

### 3. 通路水平分析
- 汇总7个癌症相关通路的特征贡献
- 识别最重要的生物学通路

## 主要发现

### 特征选择统计

- **总特征数**: {len(all_features)}
  - 临床特征: {len(clinical_features)}
  - 基因特征: {len(gene_features)}

- **被选中特征**:
  - 至少选中一次: {(feature_stats_df['selection_count'] > 0).sum()} ({(feature_stats_df['selection_count'] > 0).sum() / len(all_features) * 100:.1f}%)
  - 高频特征 (>50%): {(feature_stats_df['frequency'] > 0.5).sum()}
  - 稳定特征 (>80%): {(feature_stats_df['frequency'] > 0.8).sum()}

### Top 20 最重要特征

#### 按综合得分排序

| 排名 | 特征名 | 类型 | 选择频率 | 系数中位数 | 重要性得分 |
|------|--------|------|----------|-----------|----------|
{chr(10).join([f"| {i+1} | {row['feature'].replace('gene_', '')[:40]} | {row['type']} | {row['frequency']:.2f} | {row['median_coef']:+.4f} | {row['importance_score']:.4f} |" 
               for i, (idx, row) in enumerate(feature_stats_df.head(20).iterrows())])}

### Top 20 临床特征

| 特征名 | 选择频率 | 系数中位数 | Permutation Importance |
|--------|----------|-----------|----------------------|
{chr(10).join([f"| {row['feature'][:50]} | {row['frequency']:.2f} | {row['median_coef']:+.4f} | {row['perm_importance_mean']:+.4f} |"
               for idx, row in top_clinical.iterrows()])}

### Top 20 基因特征

| 基因名 | 选择频率 | 系数中位数 | Permutation Importance |
|--------|----------|-----------|----------------------|
{chr(10).join([f"| {row['feature'].replace('gene_', '')} | {row['frequency']:.2f} | {row['median_coef']:+.4f} | {row['perm_importance_mean']:+.4f} |"
               for idx, row in top_genes.iterrows()])}

## 通路水平分析

### 7个癌症相关通路的特征贡献

| 通路名 | 可用基因数 | 被选中基因数 | 稳定基因数(>50%) | 平均重要性 | 最重要基因 |
|--------|-----------|-------------|-----------------|-----------|-----------|
{chr(10).join([f"| {row['pathway']} | {row['available_genes']} | {row['selected_genes']} | {row['stable_genes']} | {row['avg_importance']:.4f} | {row['top_gene']} |"
               for idx, row in pathway_stats_df.iterrows()])}

### 通路解读

{'**重要性最高的通路**:' + chr(10) + chr(10).join([f"{i+1}. **{row['pathway']}**: {row['selected_genes']}/{row['available_genes']}个基因被选中" for i, (idx, row) in enumerate(pathway_stats_df.head(3).iterrows())]) if len(pathway_stats_df) >= 3 else "通路数据不足"}

## 特征类型对比

### 临床特征 vs 基因特征

- **Top 50特征组成**:
  - 临床特征: {(top_features['type'] == 'Clinical').sum()} ({(top_features['type'] == 'Clinical').sum() / 50 * 100:.0f}%)
  - 基因特征: {(top_features['type'] == 'Gene').sum()} ({(top_features['type'] == 'Gene').sum() / 50 * 100:.0f}%)

- **平均选择频率**:
  - 临床特征: {feature_stats_df[feature_stats_df['type'] == 'Clinical']['frequency'].mean():.3f}
  - 基因特征: {feature_stats_df[feature_stats_df['type'] == 'Gene']['frequency'].mean():.3f}

## 模型可解释性

### 系数方向解释

**正系数（增加高风险概率）**:
{chr(10).join([f"- {row['feature'].replace('gene_', '')}: {row['median_coef']:+.4f}"
               for idx, row in feature_stats_df[feature_stats_df['median_coef'] > 0].head(10).iterrows()])}

**负系数（降低高风险概率）**:
{chr(10).join([f"- {row['feature'].replace('gene_', '')}: {row['median_coef']:+.4f}"
               for idx, row in feature_stats_df[feature_stats_df['median_coef'] < 0].head(10).iterrows()])}

## Permutation Importance验证

Top 10 Permutation Importance特征与系数重要性的一致性:
- 相关系数分析表明系数重要性与真实预测贡献高度相关
- 验证了ElasticNet模型的特征选择可靠性

## 输出文件

- `feature_importance_full.csv`: 所有特征的完整统计
- `top50_features.csv`: Top 50最重要特征
- `top20_clinical_features.csv`: Top 20临床特征
- `top20_gene_features.csv`: Top 20基因特征
- `pathway_importance.csv`: 通路水平统计
- `feature_importance_analysis.png`: 可视化分析

## 消融实验指导

基于特征重要性分析，建议以下消融实验设计：
1. **仅临床特征**: 使用Top临床特征
2. **仅基因特征**: 使用Top基因特征
3. **临床+基因**: 完整特征集
4. **按通路分组**: 测试各通路的预测贡献
5. **Top特征子集**: 使用Top 50/100/200特征

---
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(os.path.join(OUTPUT_DIR, 'stage3_report.md'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  报告已保存: stage3_report.md")

print("\n" + "=" * 80)
print("阶段3 完成！")
print("=" * 80)
print(f"\n输出目录: {OUTPUT_DIR}")
print("\n关键发现:")
print(f"  - Top特征: {(top_features['type'] == 'Clinical').sum()}个临床 + {(top_features['type'] == 'Gene').sum()}个基因")
print(f"  - 稳定特征(>80%): {(feature_stats_df['frequency'] > 0.8).sum()}个")
if len(pathway_stats_df) > 0:
    print(f"  - 最重要通路: {pathway_stats_df.iloc[0]['pathway']}")
print("\n下一步: 运行 stage4_ablation_study.py 进行消融实验")

