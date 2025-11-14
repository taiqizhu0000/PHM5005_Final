# PHM5005 项目文件清单

## 📚 文档文件

| 文件名 | 大小 | 说明 | 重要性 |
|--------|------|------|--------|
| `README.md` | 14KB | 完整项目说明文档 | ⭐⭐⭐ |
| `QUICKSTART.md` | 3.1KB | 5分钟快速开始指南 | ⭐⭐ |
| `requirements.txt` | 594B | Python依赖包列表 | ⭐⭐⭐ |
| `PROJECT_FILES.md` | - | 本文件（项目清单） | ⭐ |

## 💻 代码文件

### 数据处理
| 文件名 | 大小 | 说明 | 运行时间 |
|--------|------|------|----------|
| `process_data.py` | 22KB | 数据清洗和特征工程 | ~2分钟 |

### 分析流程（按执行顺序）
| 文件名 | 大小 | 说明 | 运行时间 |
|--------|------|------|----------|
| `stage1_data_preparation.py` | 11KB | 数据加载和分割 | ~10秒 |
| `stage2_nested_cv_training.py` | 20KB | 嵌套CV模型训练 | ~6分钟 |
| `stage3_feature_importance.py` | 20KB | 特征重要性分析 | ~2分钟 |
| `stage4_ablation_study.py` | 15KB | 消融实验 | ~3-5分钟 |
| `stage5_final_summary.py` | 21KB | 最终报告生成 | ~10秒 |

**总运行时间**: 约15-20分钟

## 📁 数据文件

### 输入数据（dataset/）
```
dataset/
├── processed_data_phm5005.csv          [5.8MB] 完整处理后数据 ⭐
├── data_processing_documentation.md    [6.3KB] 数据文档
├── clinical_SJ_cleaned_filtered.csv    临床信息
├── case-id_map-to_rna-file-id-name.tsv  患者-RNA文件映射
├── rna-seq/                            RNA-seq表达数据
│   └── [976 files]                     各患者的基因表达文件
└── pathway_gene_list/                  通路基因列表
    ├── IGF1_signaling_symbols.csv
    ├── MAPK_hsa04010_symbols.csv
    ├── MMR_hsa03430_symbols.csv
    ├── mTOR_hsa04150_symbols.csv
    ├── p53_hsa04115_symbols.csv
    ├── PI3K_Akt_hsa04151_symbols.csv
    └── WNT_hsa04310_symbols.csv
```

### 原始数据（raw_data/）
```
raw_data/
└── TCGA-pan-cancer-clinical-data_label-data.csv  标签数据（PFI）
```

## 📊 结果文件

### 阶段1输出（results/）
```
results/
├── X_train.npy                 [2.1MB] 训练集特征
├── X_test.npy                  [546KB] 测试集特征
├── y_train.npy                 [2.3KB] 训练集标签
├── y_test.npy                  [688B]  测试集标签
├── feature_names.json          [34KB]  特征名称映射
├── data_summary.json           [306B]  数据摘要
├── train_patient_ids.csv       [3.5KB] 训练集患者ID
├── test_patient_ids.csv        [921B]  测试集患者ID
├── stage1_data_overview.png    [156KB] 数据可视化
└── stage1_report.md            [2.4KB] 阶段1报告
```

### 阶段2输出（results/stage2_nested_cv/）
```
stage2_nested_cv/
├── final_model.pkl             ⭐⭐⭐ 训练好的模型（可用于预测）
├── outer_cv_results.csv        15次CV详细结果
├── coefficient_history.json    每次训练的系数历史
├── best_params_history.csv     最佳超参数历史
├── test_predictions.csv        测试集预测结果
├── feature_selection_history.json  特征选择历史
├── test_set_results.json       测试集评估结果
├── summary.json                性能汇总
├── nested_cv_results.png       [结果可视化]
└── stage2_report.md            性能报告
```

### 阶段3输出（results/stage3_feature_importance/）
```
stage3_feature_importance/
├── feature_importance_full.csv ⭐ 所有特征的完整统计（915行）
├── top50_features.csv          ⭐ Top 50重要特征
├── top20_clinical_features.csv Top 20临床特征
├── top20_gene_features.csv     Top 20基因特征
├── pathway_importance.csv      通路重要性统计
├── feature_importance_analysis.png  [特征分析可视化]
└── stage3_report.md            特征重要性报告
```

### 阶段4输出（results/stage4_ablation/）
```
stage4_ablation/
├── ablation_results.csv        ⭐ 消融实验结果对比
├── ablation_study.png          [消融实验可视化]
└── stage4_report.md            消融实验报告
```

### 阶段5输出（results/final_summary/）
```
final_summary/
├── FINAL_REPORT.md             ⭐⭐⭐ 完整最终报告（推荐阅读）
├── PROJECT_SUMMARY.md          ⭐ 项目快速摘要
└── comprehensive_summary.png   ⭐ 综合可视化（8个子图）
```

## 📈 文件大小统计

### 代码文件
- 总代码量: ~109KB (6个Python文件)
- 平均每个脚本: ~18KB

### 数据文件
- 处理后数据: 5.8MB
- numpy数组: 2.7MB
- 总数据量: ~10MB（不含原始RNA-seq）

### 结果文件
- 文本结果: ~200KB
- 可视化图片: ~2MB
- 模型文件: 根据特征数变化

### 总项目大小
- 核心文件: ~15MB
- 含原始RNA-seq: 根据数据量

## 🔑 重要文件标记

### ⭐⭐⭐ 必读/必用
1. `README.md` - 完整项目说明
2. `requirements.txt` - 环境配置
3. `results/stage2_nested_cv/final_model.pkl` - 训练好的模型
4. `results/final_summary/FINAL_REPORT.md` - 最终报告

### ⭐⭐ 推荐查看
1. `QUICKSTART.md` - 快速上手
2. `results/final_summary/PROJECT_SUMMARY.md` - 结果摘要
3. `results/final_summary/comprehensive_summary.png` - 综合可视化
4. 各阶段的report.md - 详细分析

### ⭐ 可选查看
1. `PROJECT_FILES.md` - 本文件（项目清单）
2. `dataset/data_processing_documentation.md` - 数据文档
3. 各阶段的详细CSV结果文件

## 📝 文件命名规范

### 代码文件
- `process_data.py` - 数据处理
- `stageN_*.py` - 阶段N的分析脚本

### 数据文件
- `*.npy` - numpy数组（特征/标签）
- `*.csv` - 表格数据
- `*.json` - 配置/元数据
- `*.tsv` - 制表符分隔数据

### 报告文件
- `*_report.md` - Markdown报告
- `*_summary.md` - 摘要文档

### 可视化文件
- `*.png` - 图片（300 DPI高清）

## 🔄 文件依赖关系

```
process_data.py
    ↓ 生成
dataset/processed_data_phm5005.csv
    ↓ 输入到
stage1_data_preparation.py
    ↓ 生成
results/*.npy + feature_names.json
    ↓ 输入到
stage2_nested_cv_training.py
    ↓ 生成
results/stage2_nested_cv/*
    ↓ 输入到
stage3_feature_importance.py + stage4_ablation_study.py
    ↓ 生成
results/stage3_feature_importance/* + results/stage4_ablation/*
    ↓ 输入到
stage5_final_summary.py
    ↓ 生成
results/final_summary/*
```

## 🎯 快速定位

### 想看模型性能？
→ `results/final_summary/PROJECT_SUMMARY.md`
→ `results/stage2_nested_cv/stage2_report.md`

### 想知道重要特征？
→ `results/stage3_feature_importance/top50_features.csv`
→ `results/stage3_feature_importance/stage3_report.md`

### 想对比不同特征组？
→ `results/stage4_ablation/ablation_results.csv`
→ `results/stage4_ablation/stage4_report.md`

### 想用模型预测？
→ `results/stage2_nested_cv/final_model.pkl`
→ `README.md` (查看"模型使用示例"部分)

### 想了解完整流程？
→ `results/final_summary/FINAL_REPORT.md`
→ `README.md`

## 📞 获取帮助

1. 查看 `README.md` 的"常见问题"部分
2. 查看 `QUICKSTART.md` 的"疑难排查"部分
3. 检查各阶段的报告文件

---

**更新时间**: 2025-11-11
**项目版本**: 1.0
**文件总数**: 30+ 核心文件

