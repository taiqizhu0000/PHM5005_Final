# PHM5005 子宫内膜癌风险预测项目

## 项目简介

本项目基于TCGA-UCEC（子宫内膜癌）队列数据，整合临床信息和RNA-seq基因表达数据，使用ElasticNet Logistic Regression构建机器学习模型，预测患者2年进展风险（PFI）。

### 主要特点

- ✅ 高维小样本场景（913特征 × 348样本）
- ✅ 嵌套交叉验证确保模型稳健性
- ✅ 自动特征选择和稀疏化
- ✅ 系统性消融实验评估特征贡献
- ✅ 完整的模型可解释性分析

### 最终性能

- **Test AUROC**: 0.8553
- **Test AUPRC**: 0.6229
- **Recall**: 68.75%
- **Specificity**: 88.89%

---

## 环境配置

### 1. 创建Conda环境

```bash
conda create -n phm5005 python=3.9 -y
conda activate phm5005
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 项目结构

```
5005/
├── README.md                          # 本文件
├── requirements.txt                    # Python依赖包
│
├── dataset/                           # 数据目录
│   ├── processed_data_phm5005.csv    # 处理后的完整数据
│   ├── clinical_SJ_cleaned_filtered.csv
│   ├── case-id_map-to_rna-file-id-name.tsv
│   ├── rna-seq/                      # RNA-seq表达数据
│   └── pathway_gene_list/            # 7个通路基因列表
│
├── raw_data/                          # 原始数据
│   └── TCGA-pan-cancer-clinical-data_label-data.csv
│
├── process_data.py                    # 数据预处理脚本
├── stage1_data_preparation.py         # 阶段1: 数据准备
├── stage2_nested_cv_training.py       # 阶段2: 模型训练
├── stage3_feature_importance.py       # 阶段3: 特征重要性
├── stage4_ablation_study.py          # 阶段4: 消融实验
├── stage5_final_summary.py           # 阶段5: 最终总结
│
└── results/                          # 所有结果输出
    ├── stage1_data_overview.png
    ├── stage2_nested_cv/
    ├── stage3_feature_importance/
    ├── stage4_ablation/
    └── final_summary/
```

---

## 代码说明

### 数据处理

#### `process_data.py`

**功能**: 数据清洗、特征工程和标签生成

**输入**:
- `dataset/clinical_SJ_cleaned_filtered.csv` - 临床信息
- `dataset/rna-seq/` - RNA-seq表达数据
- `dataset/pathway_gene_list/` - 通路基因列表
- `raw_data/TCGA-pan-cancer-clinical-data_label-data.csv` - 标签数据

**输出**:
- `dataset/processed_data_phm5005.csv` - 完整处理后数据
- `dataset/data_processing_documentation.md` - 数据文档

**使用方法**:
```bash
conda activate phm5005
python process_data.py
```

**主要功能**:
1. 读取7个通路基因列表（IGF1, MAPK, MMR, mTOR, p53, PI3K-Akt, WNT）
2. 处理临床特征（标准化、One-hot编码、重分类）
   - 12个临床特征类别，生成34个特征
3. 提取RNA-seq基因表达（log2(TPM+1)转换和标准化）
   - 仅使用通路基因的并集（879个基因）
4. 根据PFI规则生成风险标签
   - 高风险: PFI=1 且 PFI.time≤730天
   - 低风险: PFI=0 且 PFI.time>730天，或 PFI=1 且 PFI.time>730天
5. 合并并保存最终数据集（348样本 × 913特征）

---

### 阶段1: 数据准备

#### `stage1_data_preparation.py`

**功能**: 数据加载、分析和分层分割

**输入**:
- `dataset/processed_data_phm5005.csv`

**输出**:
- `results/X_train.npy`, `results/X_test.npy` - 训练/测试集特征
- `results/y_train.npy`, `results/y_test.npy` - 训练/测试集标签
- `results/feature_names.json` - 特征名称映射
- `results/data_summary.json` - 数据摘要
- `results/stage1_data_overview.png` - 数据可视化
- `results/stage1_report.md` - 阶段报告

**使用方法**:
```bash
python stage1_data_preparation.py
```

**主要功能**:
1. 加载处理后的数据
2. 分析数据结构和标签分布
3. 分离临床特征（34个）和基因特征（879个）
4. 80/20分层分割（训练278 / 测试70）
5. 保存numpy格式数据供后续使用
6. 生成数据分布可视化

---

### 阶段2: 嵌套交叉验证模型训练

#### `stage2_nested_cv_training.py`

**功能**: ElasticNet Logistic Regression模型训练和评估

**输入**:
- `results/X_train.npy`, `results/y_train.npy`
- `results/X_test.npy`, `results/y_test.npy`
- `results/feature_names.json`

**输出**:
- `results/stage2_nested_cv/final_model.pkl` - 最终模型（可用于预测）⭐
- `results/stage2_nested_cv/outer_cv_results.csv` - 15次CV结果
- `results/stage2_nested_cv/coefficient_history.json` - 系数历史
- `results/stage2_nested_cv/best_params_history.csv` - 最佳参数
- `results/stage2_nested_cv/test_predictions.csv` - 测试集预测
- `results/stage2_nested_cv/nested_cv_results.png` - 结果可视化
- `results/stage2_nested_cv/stage2_report.md` - 性能报告

**使用方法**:
```bash
python stage2_nested_cv_training.py
```

**主要功能**:
1. **外层CV**: 5-Fold × 3重复 = 15次训练（评估泛化性能）
2. **内层CV**: 5-Fold GridSearchCV（超参数优化）
3. **超参数空间**:
   - `selector__k`: [300, 500, 700, all]
   - `classifier__l1_ratio`: [0.2, 0.5, 0.8]
   - `classifier__C`: [0.001, 0.01, 0.1, 1, 10, 100]
4. **Pipeline**: StandardScaler → SelectKBest → ElasticNet Logistic Regression
5. 记录每次训练的特征选择和系数
6. 测试集最终评估（AUROC 0.8553）

**训练时间**: ~6分钟

---

### 阶段3: 特征重要性分析

#### `stage3_feature_importance.py`

**功能**: 跨折稳定性分析和特征重要性排名

**输入**:
- `results/stage2_nested_cv/coefficient_history.json`
- `results/stage2_nested_cv/final_model.pkl`
- `results/X_test.npy`, `results/y_test.npy`
- `results/feature_names.json`
- `dataset/pathway_gene_list/`

**输出**:
- `results/stage3_feature_importance/feature_importance_full.csv` - 完整特征统计
- `results/stage3_feature_importance/top50_features.csv` - Top 50特征
- `results/stage3_feature_importance/top20_clinical_features.csv` - Top临床特征
- `results/stage3_feature_importance/top20_gene_features.csv` - Top基因特征
- `results/stage3_feature_importance/pathway_importance.csv` - 通路重要性
- `results/stage3_feature_importance/feature_importance_analysis.png` - 可视化
- `results/stage3_feature_importance/stage3_report.md` - 分析报告

**使用方法**:
```bash
python stage3_feature_importance.py
```

**主要功能**:
1. **跨折稳定性**: 统计15次训练中每个特征被选中的频率
2. **系数统计**: 计算非零系数的中位数、均值、四分位数
3. **综合重要性得分**: 频率 × |系数中位数|
4. **Permutation Importance**: 测试集验证真实贡献
5. **通路分析**: 7个通路的特征贡献汇总
6. Top特征排名（临床vs基因）

**关键发现**:
- 稳定特征（频率>80%）: 4个
- Top特征: 10个临床 + 40个基因
- 最重要临床特征: is_primary_disease, tumor_classification_primary

---

### 阶段4: 消融实验

#### `stage4_ablation_study.py`

**功能**: 系统评估不同特征组合的预测性能

**输入**:
- `results/X_train.npy`, `results/y_train.npy`
- `results/X_test.npy`, `results/y_test.npy`
- `results/feature_names.json`
- `results/stage3_feature_importance/feature_importance_full.csv`
- `dataset/pathway_gene_list/`

**输出**:
- `results/stage4_ablation/ablation_results.csv` - 消融实验结果
- `results/stage4_ablation/ablation_study.png` - 对比可视化
- `results/stage4_ablation/stage4_report.md` - 实验报告

**使用方法**:
```bash
python stage4_ablation_study.py
```

**主要功能**:
1. **仅临床特征** (34个) - AUROC 0.8999 ✅
2. **仅基因特征** (879个) - AUROC 0.5683
3. **临床 + 基因（完整）** (913个) - AUROC 0.8553
4. **临床 + Top 100基因** (134个) - AUROC 0.8519
5. **各通路分组测试** (7个通路)

每组使用5折交叉验证和独立测试集评估。

**训练时间**: ~3-5分钟

**关键发现**:
- 临床特征单独使用即可达到最佳性能
- 基因特征提供补充信息但单独使用效果有限
- 特征选择可在保持性能的同时降低复杂度

---

### 阶段5: 最终总结

#### `stage5_final_summary.py`

**功能**: 汇总所有结果并生成综合报告

**输入**:
- 所有前4个阶段的输出结果

**输出**:
- `results/final_summary/FINAL_REPORT.md` - 完整最终报告 ⭐⭐⭐
- `results/final_summary/PROJECT_SUMMARY.md` - 项目简要总结 ⭐
- `results/final_summary/comprehensive_summary.png` - 综合可视化 ⭐

**使用方法**:
```bash
python stage5_final_summary.py
```

**主要功能**:
1. 整合所有阶段的数据和结果
2. 生成8个子图的综合可视化：
   - 数据概览、特征组成、标签分布
   - CV vs Test性能对比
   - 混淆矩阵
   - Top 15重要特征
   - 消融实验结果
3. 生成完整的项目报告（包含方法、结果、讨论）
4. 创建项目简要总结

---

## 完整运行流程

### 方法1: 从头开始（含数据处理）

```bash
# 1. 激活环境
conda activate phm5005

# 2. 数据处理（如果已有processed_data_phm5005.csv可跳过）
python process_data.py

# 3. 依次运行5个阶段
python stage1_data_preparation.py
python stage2_nested_cv_training.py
python stage3_feature_importance.py
python stage4_ablation_study.py
python stage5_final_summary.py
```

**总运行时间**: 约15-20分钟

### 方法2: 仅运行分析流程（已有处理后数据）

```bash
conda activate phm5005

# 确保 dataset/processed_data_phm5005.csv 存在
python stage1_data_preparation.py
python stage2_nested_cv_training.py
python stage3_feature_importance.py
python stage4_ablation_study.py
python stage5_final_summary.py
```

### 方法3: 单独运行某个阶段

```bash
# 例如：只重新运行特征重要性分析
python stage3_feature_importance.py

# 或只重新生成最终报告
python stage5_final_summary.py
```

---

## 核心算法

### 模型: ElasticNet Logistic Regression

**优势**:
- L1正则化实现自动特征选择（稀疏化）
- L2正则化防止过拟合
- 适合高维小样本场景
- 模型可解释性强

**Pipeline**:
```
StandardScaler → SelectKBest(f_classif) → LogisticRegression(
    solver='saga',
    penalty='elasticnet',
    l1_ratio=0.2,
    C=0.01,
    class_weight='balanced'
)
```

### 验证策略

**嵌套交叉验证**:
- 外层: 5-Fold StratifiedKFold × 3 重复（评估性能）
- 内层: 5-Fold GridSearchCV（优化超参数）
- 独立测试集: 20% hold-out（最终评估）

---

## 主要结果文件

### 必看文件 ⭐

1. **`results/final_summary/FINAL_REPORT.md`**
   - 完整的项目报告（方法、结果、讨论）
   - 适合论文写作参考

2. **`results/final_summary/PROJECT_SUMMARY.md`**
   - 快速了解项目结果
   - 适合展示汇报

3. **`results/stage2_nested_cv/final_model.pkl`**
   - 训练好的模型
   - 可用于新数据预测

4. **`results/final_summary/comprehensive_summary.png`**
   - 一张图看懂所有结果

### 详细分析文件

- `results/stage2_nested_cv/stage2_report.md` - 模型性能详情
- `results/stage3_feature_importance/stage3_report.md` - 特征分析
- `results/stage4_ablation/stage4_report.md` - 消融实验
- 各阶段PNG文件 - 可视化结果

---

## 模型使用示例

### 加载训练好的模型进行预测

```python
import pickle
import numpy as np
import pandas as pd

# 1. 加载模型
with open('results/stage2_nested_cv/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. 准备新数据（必须与训练数据格式一致）
# X_new shape: (n_samples, 913)
# 前34列为临床特征，后879列为基因表达特征

# 3. 预测
y_pred = model.predict(X_new)  # 0或1
y_proba = model.predict_proba(X_new)[:, 1]  # 高风险概率

# 4. 解释
risk_level = ['低风险' if p < 0.5 else '高风险' for p in y_proba]
print(f"预测风险等级: {risk_level}")
print(f"高风险概率: {y_proba}")
```

---

## 数据说明

### 标签定义

**高风险 (Label = 1)**:
- PFI (进展自由生存) 事件发生
- 且事件时间 ≤ 730天（2年）

**低风险 (Label = 0)**:
- PFI无事件且时间 > 730天
- 或PFI事件发生但时间 > 730天

**排除**:
- PFI=0 且 PFI.time≤730天（随访不足）

### 特征说明

**临床特征 (34个)**:
- 人口统计学: 年龄、种族
- 肿瘤特征: FIGO分期、肿瘤分级、原发诊断
- 治疗信息: 手术、药物、放疗
- 详见 `dataset/data_processing_documentation.md`

**基因特征 (879个)**:
- 来源: 7个癌症相关KEGG通路
  - IGF1信号通路
  - MAPK信号通路
  - MMR错配修复
  - mTOR信号通路
  - p53信号通路
  - PI3K-Akt信号通路
  - Wnt信号通路
- 数值: log2(TPM+1) 标准化后的表达量

---

## 常见问题

### Q1: 运行报错 "No module named 'xxx'"?
**A**: 确保已安装所有依赖
```bash
pip install -r requirements.txt
```

### Q2: 如何只使用临床特征训练模型?
**A**: 参考阶段4的代码或直接查看消融实验结果

### Q3: 可以用自己的数据吗?
**A**: 可以，但需要确保：
1. 特征格式与训练数据一致（913个特征，相同顺序）
2. 数值特征已标准化
3. 分类特征已one-hot编码

### Q4: 训练时间太长怎么办?
**A**: 可以修改stage2中的参数：
- 减少重复次数（n_repeats=3改为1）
- 减少超参数搜索空间
- 使用n_jobs=-1并行计算

### Q5: 如何解释某个患者的预测结果?
**A**: 查看 `results/stage3_feature_importance/top50_features.csv`，观察该患者在重要特征上的取值

---

## 技术栈

- **Python**: 3.9
- **核心库**: scikit-learn, pandas, numpy
- **可视化**: matplotlib, seaborn
- **环境管理**: Conda

---

## 引用

如果您使用了本项目的代码或方法，请引用：

```
PHM5005 Group Project
Endometrial Cancer Risk Prediction Using Machine Learning
TCGA-UCEC Dataset, 2025
```

---

## 作者

PHM5005 Group Project

---

## 许可

本项目仅用于学术研究和教学目的。

---

## 致谢

- TCGA项目提供的公开数据
- KEGG通路数据库
- scikit-learn开发团队

---

## 更新日志

### 2025-11-10
- ✅ 完成数据处理和特征工程
- ✅ 实现嵌套交叉验证框架
- ✅ 完成特征重要性分析
- ✅ 完成消融实验
- ✅ 生成最终报告

---

## 联系方式

如有问题或建议，请通过课程邮件联系。

---

**📊 数据驱动，科学严谨，结果可靠！**

