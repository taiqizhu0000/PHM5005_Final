#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHM5005 Data Processing Script (Updated Version)
Integrates clinical information, RNA-seq expression data, and risk labels
Using clinical_Ju_cleaned_filtered.csv and ordinal encoding
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configure paths
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
print("PHM5005 Data Processing Started (Using New Clinical Data)")
print("=" * 80)

# ============================================================================
# Step 1: Read pathway gene list
# ============================================================================
print("\n[Step 1] Reading pathway gene list...")

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
    print(f"  - {os.path.basename(file)}: {len(genes)} genes")

print(f"  Total: {len(all_genes)} unique genes")

# ============================================================================
# Step 2: Read clinical information data
# ============================================================================
print("\n[Step 2] Reading clinical information data...")

clinical_df = pd.read_csv(CLINICAL_FILE, encoding='utf-8-sig')
print(f"  Original patient count: {len(clinical_df)}")

# Remove BOM character from first column (if exists)
clinical_df.columns = [col.replace('\ufeff', '') for col in clinical_df.columns]

# ============================================================================
# Step 3: Process clinical information features
# ============================================================================
print("\n[Step 3] Process clinical information features...")

# Create processed dataframe
processed_clinical = pd.DataFrame()
processed_clinical['patient_id'] = clinical_df['cases.submitter_id']

# Patient exclusion criteria：days_to_consent is '--'
valid_mask = clinical_df['cases.days_to_consent'] != "'--"
print(f"  Excluding patients with days_to_consent='--': {(~valid_mask).sum()} patients")

# Exclude patients with missing critical fields
age_index_valid = ~clinical_df['demographic.age_at_index'].isin(["'--", "--", ""])
age_diagnosis_valid = ~clinical_df['diagnoses.age_at_diagnosis'].isin(["'--", "--", ""])
valid_mask = valid_mask & age_index_valid & age_diagnosis_valid
print(f"  Excluding patients with missing age fields, remaining: {valid_mask.sum()} patients")

clinical_df = clinical_df[valid_mask].reset_index(drop=True)
processed_clinical = processed_clinical[valid_mask].reset_index(drop=True)

# --- 1. cases.days_to_consent (numeric, not normalized) ---
days_to_consent = pd.to_numeric(clinical_df['cases.days_to_consent'], errors='coerce')
processed_clinical['days_to_consent'] = days_to_consent

# --- 2. cases.disease_type (categorical，one-hot) ---
disease_type_map = {
    'Adenomas and Adenocarcinomas': 'Adenomas_and_Adenocarcinomas',
    'Cystic, Mucinous and Serous Neoplasms': 'Cystic_Mucinous_Serous_Neoplasms',
    'Epithelial Neoplasms, NOS': 'Epithelial_Neoplasms_NOS'
}
disease_cleaned = clinical_df['cases.disease_type'].str.strip('"').map(disease_type_map)
disease_dummies = pd.get_dummies(disease_cleaned, prefix='disease_type')
processed_clinical = pd.concat([processed_clinical, disease_dummies], axis=1)

# --- 3. demographic.race (categorical，one-hot) ---
race_cleaned = clinical_df['demographic.race'].replace(["'--", "--", ""], np.nan)
race_dummies = pd.get_dummies(race_cleaned, prefix='race', dummy_na=False)
processed_clinical = pd.concat([processed_clinical, race_dummies], axis=1)

# --- 4. demographic.age_at_index (numeric，not normalized) ---
age_at_index = pd.to_numeric(clinical_df['demographic.age_at_index'], errors='coerce')
processed_clinical['age_at_index'] = age_at_index

# --- 5. diagnoses.age_at_diagnosis (numeric，not normalized) ---
age_at_diagnosis = pd.to_numeric(clinical_df['diagnoses.age_at_diagnosis'], errors='coerce')
processed_clinical['age_at_diagnosis'] = age_at_diagnosis

# --- 6. diagnoses.classification_of_tumor (categorical，one-hot) ---
tumor_class = clinical_df['diagnoses.classification_of_tumor'].replace(["'--", "--"], np.nan)
tumor_dummies = pd.get_dummies(tumor_class, prefix='tumor_classification', dummy_na=False)
processed_clinical = pd.concat([processed_clinical, tumor_dummies], axis=1)

# --- 7. diagnoses.diagnosis_is_primary_disease (binary classification) ---
is_primary = clinical_df['diagnoses.diagnosis_is_primary_disease'].map({'TRUE': 1, 'FALSE': 0})
processed_clinical['is_primary_disease'] = is_primary

# --- 8. diagnoses.figo_stage (ordinal encoding: 0-3) ---
def encode_figo_stage(stage):
    """
    Encode FIGO stage as ordinal integer
    Stage I = 0, Stage II = 1, Stage III = 2, Stage IV = 3
    """
    if pd.isna(stage) or stage in ["'--", "--", "FALSE"]:
        return np.nan
    stage_str = str(stage).strip('"')
    
    # Stage IV (highest level)
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
print(f"  FIGO Stage encoding distribution: {figo_encoded.value_counts().sort_index()}")

# --- 9. diagnoses.primary_diagnosis (recoded into 6 categories，one-hot) ---
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
    elif 'Stage' in diag_str:  # these are misclassified stage information
        return 'Missing'
    else:
        return 'Other'

diagnosis_cat = clinical_df['diagnoses.primary_diagnosis'].apply(categorize_diagnosis)
diagnosis_dummies = pd.get_dummies(diagnosis_cat, prefix='primary_diagnosis')
processed_clinical = pd.concat([processed_clinical, diagnosis_dummies], axis=1)

# --- 10. diagnoses.prior_malignancy (binary classification, missing treated as no) ---
prior_malignancy = clinical_df['diagnoses.prior_malignancy'].replace(
    ["'--", "--", ""], 'no'
).map({'yes': 1, 'no': 0})
processed_clinical['prior_malignancy'] = prior_malignancy

# --- 11. diagnoses.tumor_grade (ordinal encoding: 0-3) ---
def encode_tumor_grade(grade):
    """
    Encode tumor grade as ordinal integer
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
print(f"  Tumor Grade encoding distribution: {tumor_grade_encoded.value_counts().sort_index()}")

# ============================================================================
# Step 4: Process label data
# ============================================================================
print("\n[Step 4] Processing label data...")

label_df = pd.read_csv(LABEL_FILE)
print(f"  Total patients in label file: {len(label_df)}")

# Apply label rules
def assign_risk_label(row):
    pfi = row['PFI']
    pfi_time = row['PFI.time']
    
    # Handle missing values
    if pd.isna(pfi) or pd.isna(pfi_time) or pfi_time == '#N/A':
        return np.nan
    
    try:
        pfi = int(pfi)
        pfi_time = float(pfi_time)
    except:
        return np.nan
    
    # High risk: PFI == 1 AND PFI.time <= 730
    if pfi == 1 and pfi_time <= 730:
        return 1
    
    # Low risk: (PFI == 0 AND PFI.time > 730) OR (PFI == 1 AND PFI.time > 730)
    if (pfi == 0 and pfi_time > 730) or (pfi == 1 and pfi_time > 730):
        return 0
    
    # Exclude: PFI == 0 AND PFI.time <= 730
    if pfi == 0 and pfi_time <= 730:
        return np.nan
    
    return np.nan

label_df['risk_label'] = label_df.apply(assign_risk_label, axis=1)
label_df = label_df[['bcr_patient_barcode', 'risk_label']].dropna()

print(f"  Valid labels count: {len(label_df)}")
print(f"    - High risk (1): {(label_df['risk_label'] == 1).sum()}")
print(f"    - Low risk (0): {(label_df['risk_label'] == 0).sum()}")

# ============================================================================
# Step 5: Process RNA-seq data
# ============================================================================
print("\n[Step 5] Step 5: Process RNA-seq data...")

# Read case-id mapping file
case_map_df = pd.read_csv(CASE_MAP_FILE, sep='\t')
print(f"  Mapping file record count: {len(case_map_df)}")

# Keep only Primary Tumor samples
primary_map = case_map_df[case_map_df['Tumor Descriptor'] == 'Primary'].copy()
print(f"  Primary Tumor sample count: {len(primary_map)}")

# Initialize RNA expression data storage
rna_expression_dict = {}
processed_patients = 0
skipped_patients = 0

for patient_id in processed_clinical['patient_id'].unique():
    # Find RNA files for this patient
    patient_files = primary_map[primary_map['Case ID'] == patient_id]
    
    if len(patient_files) == 0:
        skipped_patients += 1
        continue
    
    # If there are multiple files, read all and take average
    all_expressions = []
    
    for _, row in patient_files.iterrows():
        file_id = row['File ID']
        rna_dir = os.path.join(RNA_SEQ_DIR, file_id)
        
        if not os.path.exists(rna_dir):
            continue
        
        # Find tsv files
        tsv_files = glob.glob(os.path.join(rna_dir, "*.tsv"))
        if len(tsv_files) == 0:
            continue
        
        tsv_file = tsv_files[0]
        
        try:
            # Read RNA-seq data
            rna_df = pd.read_csv(tsv_file, sep='\t', comment='#')
            
            # Filter target genes
            rna_df = rna_df[rna_df['gene_name'].isin(all_genes)]
            
            # Extract TPM values and apply log2(TPM+1) transformation
            expression = rna_df.set_index('gene_name')['tpm_unstranded']
            expression = np.log2(expression + 1)
            
            all_expressions.append(expression)
        except Exception as e:
            continue
    
    if len(all_expressions) > 0:
        # If there are multiple samples, take average
        avg_expression = pd.concat(all_expressions, axis=1).mean(axis=1)
        rna_expression_dict[patient_id] = avg_expression
        processed_patients += 1
    else:
        skipped_patients += 1

print(f"  Successfully processed patients: {processed_patients}")
print(f"  Skipped patients: {skipped_patients}")

# Create RNA expression matrix
if len(rna_expression_dict) > 0:
    rna_df = pd.DataFrame(rna_expression_dict).T
    rna_df = rna_df.fillna(0)  # Fill missing genes with 0
    print(f"  RNA expression matrix: {rna_df.shape[0]} patients × {rna_df.shape[1]} genes")
    
    # # Do not normalize here! Normalize separately after stage1 split
    # Add gene_ prefix to avoid column name conflicts
    rna_df.columns = ['gene_' + col for col in rna_df.columns]
else:
    print("  Warning: No RNA-seq data found!")
    rna_df = pd.DataFrame()

# ============================================================================
# Step 6: Merge all data
# ============================================================================
print("\n[Step 6] Merging all data...")

# Do not normalize here! Leave it to stage1 to process training and test sets separately
# Merge clinical data and labels
merged_df = processed_clinical.merge(
    label_df, 
    left_on='patient_id', 
    right_on='bcr_patient_barcode', 
    how='inner'
)
merged_df = merged_df.drop('bcr_patient_barcode', axis=1)
print(f"  After merging clinical data and labels: {len(merged_df)} patients")

# Merge RNA data
if not rna_df.empty:
    rna_df_with_id = rna_df.reset_index().rename(columns={'index': 'patient_id'})
    final_df = merged_df.merge(rna_df_with_id, on='patient_id', how='inner')
    print(f"  After merging RNA data: {len(final_df)} patients")
else:
    final_df = merged_df
    print(f"  Warning: RNA data not merged")

# Rearrange columns: patient_id in first column, risk_label in last column
cols = ['patient_id'] + [col for col in final_df.columns if col not in ['patient_id', 'risk_label']] + ['risk_label']
final_df = final_df[cols]

print(f"\nFinal dataset: {final_df.shape[0]} patients × {final_df.shape[1]} features")
print(f"  - Clinical features count: {processed_clinical.shape[1] - 1}")
if not rna_df.empty:
    print(f"  - Gene features count: {rna_df.shape[1]}")
print(f"  - Label column: 1")

# ============================================================================
# Step 7: Save data
# ============================================================================
print("\n[Step 7] Saving processed data...")

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"  Data saved to: {OUTPUT_FILE}")

# ============================================================================
# Step 8: Generate documentation
# ============================================================================
print("\n[Step 8] Generating data documentation...")

from datetime import datetime

n_patients = len(final_df)
n_features = final_df.shape[1] - 2  # subtract patient_id and risk_label
n_clinical = processed_clinical.shape[1] - 1
n_genes = rna_df.shape[1] if not rna_df.empty else 0
n_high_risk = (final_df['risk_label'] == 1).sum()
n_low_risk = (final_df['risk_label'] == 0).sum()
pct_high_risk = 100 * n_high_risk / n_patients
pct_low_risk = 100 * n_low_risk / n_patients

doc_content = f"""# PHM5005 Data Processing Documentation (Updated Version)

## Data Overview

- **Final Patient Count**: {n_patients}
- **Total Features**: {n_features}
- **Clinical Features**: {n_clinical}
- **Gene Expression Features**: {n_genes}
- **Label**: risk_label (0=Low risk, 1=High risk)

## Label Distribution

- High risk (Label=1): {n_high_risk} ({pct_high_risk:.1f}%)
- Low risk (Label=0): {n_low_risk} ({pct_low_risk:.1f}%)

## Feature Description

### 1. Clinical Features

All clinical features source: `raw_data/clinical_Ju_cleaned_filtered.csv`

Patient unique identifier: `patient_id` (corresponds to cases.submitter_id)

#### 1. Numeric Features (not normalized, will be processed separately in stage1)

| Feature Name | Original Column | Description | Processing Method |
|--------|--------|------|----------|
| days_to_consent | cases.days_to_consent | Days to patient consent | Raw numeric value |
| age_at_index | demographic.age_at_index | Age at diagnosis (years) | Raw numeric value |
| age_at_diagnosis | diagnoses.age_at_diagnosis | Age at diagnosis (days) | Raw numeric value |

#### 2. Disease Type (One-hot encoding)

Source: `cases.disease_type`

| Feature Name | Description |
|--------|------|
| disease_type_Adenomas_and_Adenocarcinomas | Adenomas and Adenocarcinomas |
| disease_type_Cystic_Mucinous_Serous_Neoplasms | Cystic, Mucinous and Serous Neoplasms |
| disease_type_Epithelial_Neoplasms_NOS | Epithelial Neoplasms, NOS |

#### 3. Race (One-hot encoding)

Source: `demographic.race`

| Feature Name | Description |
|--------|------|
| race_white | White |
| race_black or african american | Black or African American |
| race_asian | Asian |

#### 4. Tumor Classification (One-hot encoding)

Source: `diagnoses.classification_of_tumor`

| Feature Name | Description |
|--------|------|
| tumor_classification_primary | Primary tumor |
| tumor_classification_metastasis | Metastasis |

#### 5. Primary Disease Status (Binary classification)

| Feature Name | Description | Values |
|--------|------|-----|
| is_primary_disease | Whether it is a primary disease | 1=Yes, 0=No |

#### 6. FIGO Stage (Ordinal encoding) New method

Source: `diagnoses.figo_stage`

**Encoding Rules** (ordinal integers，Stage IV > III > II > I):
- Stage I (includes IA, IB, IC) = 0
- Stage II (includes IIA, IIB) = 1
- Stage III (includes IIIA, IIIB, IIIC, IIIC1, IIIC2) = 2
- Stage IV (includes IVA, IVB) = 3

| Feature Name | Description | Value Range |
|--------|------|--------|
| figo_stage_encoded | FIGO stage ordinal encoding | 0-3 (higher is more severe) |

#### 7. Primary Diagnosis Type (One-hot encoding)

Source: `diagnoses.primary_diagnosis`

| Feature Name | Description |
|--------|------|
| primary_diagnosis_Endometrioid | Endometrioid adenocarcinoma |
| primary_diagnosis_Serous | Serous cystadenocarcinoma |
| primary_diagnosis_Clear_cell | Clear cell carcinoma |
| primary_diagnosis_Carcinoma | Other carcinoma types |
| primary_diagnosis_Other | Other diagnoses |
| primary_diagnosis_Missing | Missing data |

#### 8. Prior Malignancy History (Binary classification)

| Feature Name | Description | Value |
|--------|------|-----|
| prior_malignancy | Prior history of malignancy | 1=Yes, 0=No |

#### 9.  Tumor Grade (Ordinal encoding) New method

Source: `diagnoses.tumor_grade`

**Encoding Rules (ordinal integers，High Grade > G3 > G2 > G1):
- G1 (well differentiated) = 0
- G2 (moderately differentiated) = 1
- G3 (poorly differentiated) = 2
- High Grade (high grade) = 3

| Feature Name | Description | Value Range |
|--------|------|--------|
| tumor_grade_encoded | Tumor grade ordinal encoding | 0-3 (higher is more malignant) |

### 2. Gene Expression Features

Source: RNA-seq data in `dataset/rna-seq/` directory

**Data Linking**:
1. Use `patient_id` to search in `dataset/case-id_map-to_rna-file-id-name.tsv`
2. Match "Case ID" column to obtain "File ID"
3. Find corresponding .tsv file in `dataset/rna-seq/{{File ID}}/` directory
4. Use only samples with Tumor Descriptor = "Primary"

**Gene Filtering**:
- Obtain gene list from all CSV files in `dataset/pathway_gene_list/`
- Union of gene names from 7 pathway files
- Total of {n_genes} genes

***Data Processing**:
- Use TPM values from `tpm_unstranded` column
- Apply log2(TPM + 1) transformation
- **Not normalized here** - Will normalize training/test sets separately in stage1

All gene feature column names format: `gene_{{GENE_SYMBOL}}`

### 3. Label

Source: `dataset/TCGA-pan-cancer-clinical-data_label-data.csv`

**Mapping**: Use `patient_id` to match `bcr_patient_barcode`

**Label Rules**:
- **High risk (Label=1)**: PFI == 1 AND PFI.time ≤ 730 days
- **Low risk (Label=0)**: (PFI == 0 AND PFI.time > 730 days) OR (PFI == 1 AND PFI.time > 730 days)
- **Exclude**: PFI == 0 AND PFI.time ≤ 730 days

| Feature Name | Description | Values |
|--------|------|-----|
| risk_label | Patient risk level | 1=High risk, 0=Low risk |

## Data Exclusion Criteria

The following patients were excluded:
1. Patients with `cases.days_to_consent` as '--'
2. Patients with missing critical age fields
3. Patients not found in RNA-seq mapping file
4. Patients without Primary Tumor samples
5. Patients not matched in label data or not meeting label criteria

## Data Quality Notes

- Numeric features retain original scale (will be normalized in stage1)
- Categorical features use one-hot encoding
- Ordinal categorical features use integer encoding (FIGO stage, Tumor grade)
- RNA expression log2 transformed (not normalized)
- No missing values (processed or excluded)

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Processing Script: process_data.py (Updated Version)
"""

with open(DOC_FILE, 'w', encoding='utf-8') as f:
    f.write(doc_content)

print(f"  Documentation saved to: {DOC_FILE}")

print("\n" + "=" * 80)
print("Data Processing Complete!")
print("=" * 80)
print(f"\nOutput Files:")
print(f"  1. {OUTPUT_FILE}")
print(f"  2. {DOC_FILE}")
print(f"\nFinal Dataset Statistics:")
print(f"  - Patient count: {n_patients}")
print(f"  - Total features: {n_features}")
print(f"  - High risk patients: {n_high_risk} ({pct_high_risk:.1f}%)")
print(f"  - Low risk patients: {n_low_risk} ({pct_low_risk:.1f}%)")
print(f"\n  Note: Numeric features not normalized, will be processed separately for training/test sets in stage1")
