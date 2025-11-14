# PHM5005 数据处理文档 (更新版)

## 数据概览

- **最终患者数**: 362
- **总特征数**: 907
- **临床特征数**: 28
- **基因表达特征数**: 875
- **标签**: risk_label (0=低风险, 1=高风险)

## 标签分布

- 高风险 (Label=1): 85 (23.5%)
- 低风险 (Label=0): 277 (76.5%)

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
3. 在 `dataset/rna-seq/{File ID}/` 目录下找到对应的 .tsv 文件
4. 仅使用 Tumor Descriptor = "Primary" 的样本

**基因筛选**:
- 从 `dataset/pathway_gene_list/` 下所有CSV文件获取基因列表
- 7个通路文件的基因名称并集
- 总计 875 个基因

**数据处理**:
- 使用 `tpm_unstranded` 列的TPM值
- 应用 log2(TPM + 1) 转换
- ⚠️ **未在此处标准化** - 将在stage1中分别对训练/测试集标准化

所有基因特征列名格式: `gene_{GENE_SYMBOL}`

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
生成时间: 2025-11-11 22:39:07
处理脚本: process_data.py (更新版)
