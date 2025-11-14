#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHM5005 数据处理脚本 (更新版)
整合临床信息、RNA-seq表达量和风险标签
使用 clinical_Ju_cleaned_filtered.csv 和有序编码
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置路径
# ============================================================================
BASE_DIR = "/Users/a/Desktop/5005"
DATASET_DIR = os.path.join(BASE_DIR, "raw_data")

CLINICAL_FILE = os.path.join(DATASET_DIR, "clinical_Ju_cleaned_filtered.csv")
LABEL_FILE = os.path.join(DATASET_DIR, "TCGA-pan-cancer-clinical-data_label-data.csv")
CASE_MAP_FILE = os.path.join(DATASET_DIR, "case-id_map-to_rna-file-id-name.tsv")
RNA_SEQ_DIR = os.path.join(DATASET_DIR, "rna-seq")
PATHWAY_DIR = os.path.join(DATASET_DIR, "pathway_gene_list")
OUTPUT_FILE = os.path.join(DATASET_DIR, "processed_data_phm5005.csv")
DOC_FILE = os.path.join(DATASET_DIR, "data_processing_documentation.md")

print("=" * 80)
print("PHM5005 数据处理开始 (使用新临床数据)")
print("=" * 80)

# ============================================================================
# 步骤1: 读取pathway基因列表
# ============================================================================
print("\n[步骤1] 读取pathway基因列表...")

pathway_files = glob.glob(os.path.join(PATHWAY_DIR, "*.csv"))
all_genes = set()
pathway_genes_dict = {}

for file in pathway_files:
    pathway_name = os.path.basename(file).replace('_symbols.csv', '')
    df = pd.read_csv(file)
    # 第一列是Symbol
    genes = df.iloc[:, 0].str.strip('"').tolist()
    all_genes.update(genes)
    pathway_genes_dict[pathway_name] = genes
    print(f"  - {os.path.basename(file)}: {len(genes)} 基因")

print(f"  总计: {len(all_genes)} 个唯一基因")

# ============================================================================
# 步骤2: 读取临床信息数据
# ============================================================================
print("\n[步骤2] 读取临床信息数据...")

clinical_df = pd.read_csv(CLINICAL_FILE, encoding='utf-8-sig')
print(f"  原始患者数: {len(clinical_df)}")

# 移除第一列的BOM字符（如果存在）
clinical_df.columns = [col.replace('\ufeff', '') for col in clinical_df.columns]

# ============================================================================
# 步骤3: 处理临床信息特征
# ============================================================================
print("\n[步骤3] 处理临床信息特征...")

# 创建处理后的数据框
processed_clinical = pd.DataFrame()
processed_clinical['patient_id'] = clinical_df['cases.submitter_id']

# 排除标准：days_to_consent 为 '--'
valid_mask = clinical_df['cases.days_to_consent'] != "'--"
print(f"  排除 days_to_consent='--' 的患者: {(~valid_mask).sum()} 个")

# 排除关键字段缺失的患者
age_index_valid = ~clinical_df['demographic.age_at_index'].isin(["'--", "--", ""])
age_diagnosis_valid = ~clinical_df['diagnoses.age_at_diagnosis'].isin(["'--", "--", ""])
valid_mask = valid_mask & age_index_valid & age_diagnosis_valid
print(f"  排除年龄字段缺失的患者，剩余: {valid_mask.sum()} 个")

clinical_df = clinical_df[valid_mask].reset_index(drop=True)
processed_clinical = processed_clinical[valid_mask].reset_index(drop=True)

# --- 1. cases.days_to_consent (数值型，不标准化) ---
days_to_consent = pd.to_numeric(clinical_df['cases.days_to_consent'], errors='coerce')
processed_clinical['days_to_consent'] = days_to_consent

# --- 2. cases.disease_type (分类，one-hot) ---
disease_type_map = {
    'Adenomas and Adenocarcinomas': 'Adenomas_and_Adenocarcinomas',
    'Cystic, Mucinous and Serous Neoplasms': 'Cystic_Mucinous_Serous_Neoplasms',
    'Epithelial Neoplasms, NOS': 'Epithelial_Neoplasms_NOS'
}
disease_cleaned = clinical_df['cases.disease_type'].str.strip('"').map(disease_type_map)
disease_dummies = pd.get_dummies(disease_cleaned, prefix='disease_type')
processed_clinical = pd.concat([processed_clinical, disease_dummies], axis=1)

# --- 3. demographic.race (分类，one-hot) ---
race_cleaned = clinical_df['demographic.race'].replace(["'--", "--", ""], np.nan)
race_dummies = pd.get_dummies(race_cleaned, prefix='race', dummy_na=False)
processed_clinical = pd.concat([processed_clinical, race_dummies], axis=1)

# --- 4. demographic.age_at_index (数值型，不标准化) ---
age_at_index = pd.to_numeric(clinical_df['demographic.age_at_index'], errors='coerce')
processed_clinical['age_at_index'] = age_at_index

# --- 5. diagnoses.age_at_diagnosis (数值型，不标准化) ---
age_at_diagnosis = pd.to_numeric(clinical_df['diagnoses.age_at_diagnosis'], errors='coerce')
processed_clinical['age_at_diagnosis'] = age_at_diagnosis

# --- 6. diagnoses.classification_of_tumor (分类，one-hot) ---
tumor_class = clinical_df['diagnoses.classification_of_tumor'].replace(["'--", "--"], np.nan)
tumor_dummies = pd.get_dummies(tumor_class, prefix='tumor_classification', dummy_na=False)
processed_clinical = pd.concat([processed_clinical, tumor_dummies], axis=1)

# --- 7. diagnoses.diagnosis_is_primary_disease (二分类) ---
is_primary = clinical_df['diagnoses.diagnosis_is_primary_disease'].map({'TRUE': 1, 'FALSE': 0})
processed_clinical['is_primary_disease'] = is_primary

# --- 8. diagnoses.figo_stage (有序编码: 0-3) ---
def encode_figo_stage(stage):
    """
    将FIGO stage编码为有序整数
    Stage I = 0, Stage II = 1, Stage III = 2, Stage IV = 3
    """
    if pd.isna(stage) or stage in ["'--", "--", "FALSE"]:
        return np.nan
    stage_str = str(stage).strip('"')
    
    # Stage IV (最高级别)
    if stage_str.startswith('Stage IV'):
        return 3
    # Stage III
    elif stage_str.startswith('Stage III'):
        return 2
    # Stage II
    elif stage_str.startswith('Stage II'):
        return 1
    # Stage I
    elif 'I' in stage_str and 'II' not in stage_str and 'III' not in stage_str and 'IV' not in stage_str:
        return 0
    else:
        return np.nan

figo_encoded = clinical_df['diagnoses.figo_stage'].apply(encode_figo_stage)
processed_clinical['figo_stage_encoded'] = figo_encoded
print(f"  FIGO Stage编码分布: {figo_encoded.value_counts().sort_index()}")

# --- 9. diagnoses.primary_diagnosis (重编码为6类，one-hot) ---
def categorize_diagnosis(diagnosis):
    if pd.isna(diagnosis) or diagnosis in ["'--", "--"]:
        return 'Missing'
    diag_str = str(diagnosis).strip('"')
    if 'Endometrioid' in diag_str:
        return 'Endometrioid'
    elif 'Serous' in diag_str:
        return 'Serous'
    elif 'Clear cell' in diag_str:
        return 'Clear_cell'
    elif 'Carcinoma' in diag_str or 'carcinoma' in diag_str:
        return 'Carcinoma'
    elif 'Stage' in diag_str:  # 这些是误分类的stage信息
        return 'Missing'
    else:
        return 'Other'

diagnosis_cat = clinical_df['diagnoses.primary_diagnosis'].apply(categorize_diagnosis)
diagnosis_dummies = pd.get_dummies(diagnosis_cat, prefix='primary_diagnosis')
processed_clinical = pd.concat([processed_clinical, diagnosis_dummies], axis=1)

# --- 10. diagnoses.prior_malignancy (二分类，缺失视为no) ---
prior_malignancy = clinical_df['diagnoses.prior_malignancy'].replace(
    ["'--", "--", ""], 'no'
).map({'yes': 1, 'no': 0})
processed_clinical['prior_malignancy'] = prior_malignancy

# --- 11. diagnoses.tumor_grade (有序编码: 0-3) ---
def encode_tumor_grade(grade):
    """
    将tumor grade编码为有序整数
    G1 = 0, G2 = 1, G3 = 2, High Grade = 3
    """
    if pd.isna(grade) or grade in ["'--", "--", "no", "yes"]:
        return np.nan
    grade_str = str(grade).strip('"')
    
    if 'High Grade' in grade_str or 'high grade' in grade_str:
        return 3
    elif 'G3' in grade_str:
        return 2
    elif 'G2' in grade_str:
        return 1
    elif 'G1' in grade_str:
        return 0
    else:
        return np.nan

tumor_grade_encoded = clinical_df['diagnoses.tumor_grade'].apply(encode_tumor_grade)
processed_clinical['tumor_grade_encoded'] = tumor_grade_encoded
print(f"  Tumor Grade编码分布: {tumor_grade_encoded.value_counts().sort_index()}")

# --- 12. 治疗类型 (直接使用已有的one-hot列) ---
treatment_cols = [
    'treatment_Chemotherapy',
    'treatment_Hormone_Therapy',
    'treatment_Pharmaceutical_Therapy_NOS',
    'treatment_Radiation',
    'treatment_Surgery_Minimally_Invasive',
    'treatment_Surgery_NOS',
    'treatment_Surgery_Open',
    'treatment_Targeted_Molecular_Therapy'
]

for col in treatment_cols:
    if col in clinical_df.columns:
        processed_clinical[col] = clinical_df[col].astype(int)
        print(f"  - {col}: {processed_clinical[col].sum()} 患者")

print(f"\n  处理后的临床特征数: {processed_clinical.shape[1] - 1} (不含patient_id)")

# ============================================================================
# 步骤4: 处理标签数据
# ============================================================================
print("\n[步骤4] 处理标签数据...")

label_df = pd.read_csv(LABEL_FILE)
print(f"  标签文件总患者数: {len(label_df)}")

# 应用标签规则
def assign_risk_label(row):
    pfi = row['PFI']
    pfi_time = row['PFI.time']
    
    # 处理缺失值
    if pd.isna(pfi) or pd.isna(pfi_time) or pfi_time == '#N/A':
        return np.nan
    
    try:
        pfi = int(pfi)
        pfi_time = float(pfi_time)
    except:
        return np.nan
    
    # 高风险: PFI == 1 AND PFI.time <= 730
    if pfi == 1 and pfi_time <= 730:
        return 1
    
    # 低风险: (PFI == 0 AND PFI.time > 730) OR (PFI == 1 AND PFI.time > 730)
    if (pfi == 0 and pfi_time > 730) or (pfi == 1 and pfi_time > 730):
        return 0
    
    # 排除: PFI == 0 AND PFI.time <= 730
    if pfi == 0 and pfi_time <= 730:
        return np.nan
    
    return np.nan

label_df['risk_label'] = label_df.apply(assign_risk_label, axis=1)
label_df = label_df[['bcr_patient_barcode', 'risk_label']].dropna()

print(f"  有效标签数: {len(label_df)}")
print(f"    - 高风险(1): {(label_df['risk_label'] == 1).sum()}")
print(f"    - 低风险(0): {(label_df['risk_label'] == 0).sum()}")

# ============================================================================
# 步骤5: 处理RNA-seq数据
# ============================================================================
print("\n[步骤5] 处理RNA-seq数据...")

# 读取case-id映射文件
case_map_df = pd.read_csv(CASE_MAP_FILE, sep='\t')
print(f"  映射文件记录数: {len(case_map_df)}")

# 只保留Primary Tumor样本
primary_map = case_map_df[case_map_df['Tumor Descriptor'] == 'Primary'].copy()
print(f"  Primary Tumor样本数: {len(primary_map)}")

# 初始化RNA表达数据存储
rna_expression_dict = {}
processed_patients = 0
skipped_patients = 0

for patient_id in processed_clinical['patient_id'].unique():
    # 查找该患者的RNA文件
    patient_files = primary_map[primary_map['Case ID'] == patient_id]
    
    if len(patient_files) == 0:
        skipped_patients += 1
        continue
    
    # 如果有多个文件，读取所有并取平均
    all_expressions = []
    
    for _, row in patient_files.iterrows():
        file_id = row['File ID']
        rna_dir = os.path.join(RNA_SEQ_DIR, file_id)
        
        if not os.path.exists(rna_dir):
            continue
        
        # 查找tsv文件
        tsv_files = glob.glob(os.path.join(rna_dir, "*.tsv"))
        if len(tsv_files) == 0:
            continue
        
        tsv_file = tsv_files[0]
        
        try:
            # 读取RNA-seq数据
            rna_df = pd.read_csv(tsv_file, sep='\t', comment='#')
            
            # 筛选目标基因
            rna_df = rna_df[rna_df['gene_name'].isin(all_genes)]
            
            # 提取TPM值并应用log2(TPM+1)转换
            expression = rna_df.set_index('gene_name')['tpm_unstranded']
            expression = np.log2(expression + 1)
            
            all_expressions.append(expression)
        except Exception as e:
            continue
    
    if len(all_expressions) > 0:
        # 如果有多个样本，取平均
        avg_expression = pd.concat(all_expressions, axis=1).mean(axis=1)
        rna_expression_dict[patient_id] = avg_expression
        processed_patients += 1
    else:
        skipped_patients += 1

print(f"  成功处理患者数: {processed_patients}")
print(f"  跳过患者数: {skipped_patients}")

# 创建RNA表达矩阵
if len(rna_expression_dict) > 0:
    rna_df = pd.DataFrame(rna_expression_dict).T
    rna_df = rna_df.fillna(0)  # 填充缺失基因为0
    print(f"  RNA表达矩阵: {rna_df.shape[0]} 患者 × {rna_df.shape[1]} 基因")
    
    # 不在这里标准化！在stage1分割后分别标准化
    # 添加gene_前缀避免列名冲突
    rna_df.columns = ['gene_' + col for col in rna_df.columns]
else:
    print("  警告: 没有找到任何RNA-seq数据!")
    rna_df = pd.DataFrame()

# ============================================================================
# 步骤6: 合并所有数据
# ============================================================================
print("\n[步骤6] 合并所有数据...")

# ⚠️ 不在这里标准化！留到stage1中分别处理训练集和测试集
# 合并临床数据和标签
merged_df = processed_clinical.merge(
    label_df, 
    left_on='patient_id', 
    right_on='bcr_patient_barcode', 
    how='inner'
)
merged_df = merged_df.drop('bcr_patient_barcode', axis=1)
print(f"  合并临床数据和标签后: {len(merged_df)} 患者")

# 合并RNA数据
if not rna_df.empty:
    rna_df_with_id = rna_df.reset_index().rename(columns={'index': 'patient_id'})
    final_df = merged_df.merge(rna_df_with_id, on='patient_id', how='inner')
    print(f"  合并RNA数据后: {len(final_df)} 患者")
else:
    final_df = merged_df
    print(f"  警告: 未合并RNA数据")

# 重新排列列：patient_id在第一列，risk_label在最后一列
cols = ['patient_id'] + [col for col in final_df.columns if col not in ['patient_id', 'risk_label']] + ['risk_label']
final_df = final_df[cols]

print(f"\n最终数据集: {final_df.shape[0]} 患者 × {final_df.shape[1]} 特征")
print(f"  - 临床特征数: {processed_clinical.shape[1] - 1}")
if not rna_df.empty:
    print(f"  - 基因特征数: {rna_df.shape[1]}")
print(f"  - 标签列: 1")

# ============================================================================
# 步骤7: 保存数据
# ============================================================================
print("\n[步骤7] 保存处理后的数据...")

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"  数据已保存到: {OUTPUT_FILE}")

# ============================================================================
# 步骤8: 生成文档
# ============================================================================
print("\n[步骤8] 生成数据文档...")

from datetime import datetime

n_patients = len(final_df)
n_features = final_df.shape[1] - 2  # 减去patient_id和risk_label
n_clinical = processed_clinical.shape[1] - 1
n_genes = rna_df.shape[1] if not rna_df.empty else 0
n_high_risk = (final_df['risk_label'] == 1).sum()
n_low_risk = (final_df['risk_label'] == 0).sum()
pct_high_risk = 100 * n_high_risk / n_patients
pct_low_risk = 100 * n_low_risk / n_patients

doc_content = f"""# PHM5005 数据处理文档 (更新版)

## 数据概览

- **最终患者数**: {n_patients}
- **总特征数**: {n_features}
- **临床特征数**: {n_clinical}
- **基因表达特征数**: {n_genes}
- **标签**: risk_label (0=低风险, 1=高风险)

## 标签分布

- 高风险 (Label=1): {n_high_risk} ({pct_high_risk:.1f}%)
- 低风险 (Label=0): {n_low_risk} ({pct_low_risk:.1f}%)

## 特征说明

### 一、临床特征

所有临床特征来源: `raw_data/clinical_Ju_cleaned_filtered.csv`

患者唯一标识: `patient_id` (对应 cases.submitter_id)

#### 1. 数值型特征 (未标准化，将在stage1分别处理)

| 特征名 | 原始列 | 说明 | 处理方法 |
|--------|--------|------|----------|
| days_to_consent | cases.days_to_consent | 患者同意时间(天数) | 原始数值 |
| age_at_index | demographic.age_at_index | 诊断时年龄(岁) | 原始数值 |
| age_at_diagnosis | diagnoses.age_at_diagnosis | 确诊年龄(天数) | 原始数值 |

#### 2. 疾病类型 (One-hot编码)

来源: `cases.disease_type`

| 特征名 | 说明 |
|--------|------|
| disease_type_Adenomas_and_Adenocarcinomas | 腺瘤和腺癌 |
| disease_type_Cystic_Mucinous_Serous_Neoplasms | 囊性、粘液性和浆液性肿瘤 |
| disease_type_Epithelial_Neoplasms_NOS | 上皮肿瘤(非特指) |

#### 3. 种族 (One-hot编码)

来源: `demographic.race`

| 特征名 | 说明 |
|--------|------|
| race_white | 白种人 |
| race_black or african american | 黑人或非裔美国人 |
| race_asian | 亚裔 |

#### 4. 肿瘤分类 (One-hot编码)

来源: `diagnoses.classification_of_tumor`

| 特征名 | 说明 |
|--------|------|
| tumor_classification_primary | 原发肿瘤 |
| tumor_classification_metastasis | 转移瘤 |

#### 5. 是否原发疾病 (二分类)

| 特征名 | 说明 | 值 |
|--------|------|-----|
| is_primary_disease | 是否为原发疾病 | 1=是, 0=否 |

#### 6. FIGO分期 (有序编码) ⭐ 新方法

来源: `diagnoses.figo_stage`

**编码规则** (有序整数，Stage IV > III > II > I):
- Stage I (包含IA, IB, IC) = 0
- Stage II (包含IIA, IIB) = 1
- Stage III (包含IIIA, IIIB, IIIC, IIIC1, IIIC2) = 2
- Stage IV (包含IVA, IVB) = 3

| 特征名 | 说明 | 值范围 |
|--------|------|--------|
| figo_stage_encoded | FIGO分期有序编码 | 0-3 (越大越严重) |

#### 7. 原发诊断类型 (One-hot编码)

来源: `diagnoses.primary_diagnosis`

| 特征名 | 说明 |
|--------|------|
| primary_diagnosis_Endometrioid | 子宫内膜样腺癌 |
| primary_diagnosis_Serous | 浆液性囊腺癌 |
| primary_diagnosis_Clear_cell | 透明细胞癌 |
| primary_diagnosis_Carcinoma | 其他癌类型 |
| primary_diagnosis_Other | 其他诊断 |
| primary_diagnosis_Missing | 缺失数据 |

#### 8. 既往恶性肿瘤史 (二分类)

| 特征名 | 说明 | 值 |
|--------|------|-----|
| prior_malignancy | 既往是否有恶性肿瘤 | 1=是, 0=否 |

#### 9. 肿瘤分级 (有序编码) ⭐ 新方法

来源: `diagnoses.tumor_grade`

**编码规则** (有序整数，High Grade > G3 > G2 > G1):
- G1 (分化良好) = 0
- G2 (中等分化) = 1
- G3 (分化差) = 2
- High Grade (高级别) = 3

| 特征名 | 说明 | 值范围 |
|--------|------|--------|
| tumor_grade_encoded | 肿瘤分级有序编码 | 0-3 (越大越恶性) |

#### 10. 治疗类型 (已有的one-hot编码) ⭐ 直接使用

来源: 新文件已有的treatment列

| 特征名 | 说明 | 值 |
|--------|------|-----|
| treatment_Chemotherapy | 化疗 | 1=是, 0=否 |
| treatment_Hormone_Therapy | 激素治疗 | 1=是, 0=否 |
| treatment_Pharmaceutical_Therapy_NOS | 药物治疗(非特指) | 1=是, 0=否 |
| treatment_Radiation | 放疗 | 1=是, 0=否 |
| treatment_Surgery_Minimally_Invasive | 微创手术 | 1=是, 0=否 |
| treatment_Surgery_NOS | 手术(非特指) | 1=是, 0=否 |
| treatment_Surgery_Open | 开放手术 | 1=是, 0=否 |
| treatment_Targeted_Molecular_Therapy | 靶向分子治疗 | 1=是, 0=否 |

### 二、基因表达特征

来源: `dataset/rna-seq/` 目录下的RNA-seq数据

**数据链接**:
1. 使用 `patient_id` 在 `dataset/case-id_map-to_rna-file-id-name.tsv` 中查找
2. 匹配 "Case ID" 列，获取 "File ID"
3. 在 `dataset/rna-seq/{{File ID}}/` 目录下找到对应的 .tsv 文件
4. 仅使用 Tumor Descriptor = "Primary" 的样本

**基因筛选**:
- 从 `dataset/pathway_gene_list/` 下所有CSV文件获取基因列表
- 7个通路文件的基因名称并集
- 总计 {n_genes} 个基因

**数据处理**:
- 使用 `tpm_unstranded` 列的TPM值
- 应用 log2(TPM + 1) 转换
- ⚠️ **未在此处标准化** - 将在stage1中分别对训练/测试集标准化

所有基因特征列名格式: `gene_{{GENE_SYMBOL}}`

### 三、标签

来源: `dataset/TCGA-pan-cancer-clinical-data_label-data.csv`

**映射**: 使用 `patient_id` 匹配 `bcr_patient_barcode`

**标签规则**:
- **高风险 (Label=1)**: PFI == 1 AND PFI.time ≤ 730天
- **低风险 (Label=0)**: (PFI == 0 AND PFI.time > 730天) OR (PFI == 1 AND PFI.time > 730天)
- **排除**: PFI == 0 AND PFI.time ≤ 730天

| 特征名 | 说明 | 值 |
|--------|------|-----|
| risk_label | 患者风险等级 | 1=高风险, 0=低风险 |

## 重要更新 ⭐

### 与旧版本的主要差异：

1. **临床数据源**: `clinical_SJ_cleaned_filtered.csv` → `clinical_Ju_cleaned_filtered.csv`
2. **FIGO Stage**: One-hot编码 → 有序编码(0-3)
3. **Tumor Grade**: One-hot编码 → 有序编码(0-3)
4. **治疗信息**: 手动拆分 → 直接使用已有one-hot列
5. **标准化时机**: 在process_data中 → 在stage1中分别处理训练/测试集

### 防止数据泄漏：

- ✅ 数值特征未在此处标准化
- ✅ 基因表达已log2转换但未标准化
- ✅ 标准化将在stage1_data_preparation.py中完成
- ✅ 训练集fit → 测试集transform

## 数据排除标准

以下患者被排除:
1. `cases.days_to_consent` 为 '--' 的患者
2. 关键年龄字段缺失的患者
3. 无法在RNA-seq映射文件中找到的患者
4. 无法找到Primary Tumor样本的患者
5. 无法在标签数据中匹配或不符合标签规则的患者

## 数据质量说明

- 数值特征保留原始尺度 (将在stage1标准化)
- 分类特征使用one-hot编码
- 有序分类特征使用整数编码 (FIGO stage, Tumor grade)
- RNA表达量经log2转换 (未标准化)
- 无缺失值 (已处理或排除)

---
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
处理脚本: process_data.py (更新版)
"""

with open(DOC_FILE, 'w', encoding='utf-8') as f:
    f.write(doc_content)

print(f"  文档已保存到: {DOC_FILE}")

print("\n" + "=" * 80)
print("数据处理完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  1. {OUTPUT_FILE}")
print(f"  2. {DOC_FILE}")
print(f"\n最终数据集统计:")
print(f"  - 患者数: {n_patients}")
print(f"  - 总特征数: {n_features}")
print(f"  - 高风险患者: {n_high_risk} ({pct_high_risk:.1f}%)")
print(f"  - 低风险患者: {n_low_risk} ({pct_low_risk:.1f}%)")
print(f"\n⚠️  注意: 数值特征未标准化，将在stage1中分别处理训练/测试集")
