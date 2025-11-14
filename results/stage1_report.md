# 阶段1: 数据准备和分析 (更新版)

## 数据加载

- 数据文件: `D:/PHM5005/5005-main\dataset\processed_data_phm5005.csv`
- 总样本数: 362
- 总特征数: 898
  - 临床特征: 19
  - 基因表达特征: 879

## 标签分布

### 总体
- 低风险 (0): 277 (76.52%)
- 高风险 (1): 85 (23.48%)

### 训练集 (289 样本)
- 低风险 (0): 221 (76.47%)
- 高风险 (1): 68 (23.53%)

### 测试集 (73 样本)
- 低风险 (0): 56 (76.71%)
- 高风险 (1): 17 (23.29%)

## 数据分割

- 分割比例: 80% 训练, 20% 测试
- 分割方法: 分层分割 (stratified split)
- 随机种子: 42

## 特征标准化 ⭐ 防止数据泄漏

### 标准化策略
1. **数值型临床特征** (4 个):
   - days_to_consent
   - age_at_index
   - age_at_diagnosis
   - figo_stage_encoded
   - tumor_grade_encoded

2. **基因表达特征** (879 个):
   - 所有 gene_* 特征

### 标准化方法
- 使用 StandardScaler (均值=0, 标准差=1)
- **关键**: 在训练集上 fit，然后 transform 测试集
- **避免数据泄漏**: 测试集信息未用于计算训练集的均值和标准差

### 保存的文件
- `scaler_clinical.pkl`: 临床特征的StandardScaler
- `scaler_gene.pkl`: 基因特征的StandardScaler

## 输出文件

### NumPy数组
- `X_train.npy`: 训练集特征 (289, 898)
- `X_test.npy`: 测试集特征 (73, 898)
- `y_train.npy`: 训练集标签 (289,)
- `y_test.npy`: 测试集标签 (73,)

### 元数据
- `feature_names.json`: 特征名称和分类
- `train_patient_ids.csv`: 训练集患者ID
- `test_patient_ids.csv`: 测试集患者ID
- `data_summary.json`: 数据统计摘要
- `scaler_clinical.pkl`: 临床特征标准化器
- `scaler_gene.pkl`: 基因特征标准化器

### 可视化
- `stage1_data_overview.png`: 数据分布可视化

## 数据质量检查

- ✅ 无缺失值
- ✅ 训练/测试集标签分布保持一致
- ✅ 特征已正确标准化
- ✅ 数据泄漏已防止

## 重要更新 ⭐

与旧版本的主要差异：
1. **FIGO Stage**: One-hot编码 → 有序编码 (0-3)
2. **Tumor Grade**: One-hot编码 → 有序编码 (0-3)
3. **标准化时机**: process_data中 → stage1中分别处理
4. **防止数据泄漏**: 训练集fit → 测试集transform

---
生成时间: 2025-11-14 17:12:26
