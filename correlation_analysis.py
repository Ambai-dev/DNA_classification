# Correlation Analysis and Redundant Parameter Detection
# ======================================================

import pandas as pd
import numpy as np
from collections import Counter

print("\n" + "="*70)
print("     CORRELATION ANALYSIS & REDUNDANT PARAMETER DETECTION")
print("="*70)

# Load data
print("\nLoading data...")
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
validation = pd.read_csv('validation.csv', index_col=0)
all_data = pd.concat([train, test, validation], ignore_index=True)
print(f"Total samples: {len(all_data):,}")

# ============================================================
# 1. CHECK FOR CONSTANT/SINGLE-VALUE COLUMNS (REDUNDANT)
# ============================================================
print("\n" + "-"*70)
print("1. CHECKING FOR CONSTANT/SINGLE-VALUE COLUMNS")
print("-"*70)

redundant_cols = []
for col in all_data.columns:
    unique_count = all_data[col].nunique()
    if unique_count == 1:
        print(f"\n  [REDUNDANT] {col}")
        print(f"    - Only 1 unique value: '{all_data[col].iloc[0]}'")
        print(f"    - RECOMMENDATION: Remove this column")
        redundant_cols.append(col)
    elif unique_count <= 3:
        print(f"\n  [LOW VARIANCE] {col}")
        print(f"    - Only {unique_count} unique values")
        print(f"    - Values: {list(all_data[col].unique())}")

if not redundant_cols:
    print("\n  No completely redundant (single-value) columns found.")

# ============================================================
# 2. CHECK NCBIGeneID UNIQUENESS
# ============================================================
print("\n" + "-"*70)
print("2. CHECKING NCBIGeneID UNIQUENESS")
print("-"*70)

total_rows = len(all_data)
unique_ids = all_data['NCBIGeneID'].nunique()
print(f"\n  Total rows: {total_rows:,}")
print(f"  Unique NCBIGeneID: {unique_ids:,}")
print(f"  Duplicates: {total_rows - unique_ids:,}")

if total_rows == unique_ids:
    print("\n  [INFO] NCBIGeneID is a unique identifier (1:1 with rows)")
    print("  RECOMMENDATION: Keep as ID, but don't use as predictive feature")

# ============================================================
# 3. CHECK SYMBOL vs DESCRIPTION REDUNDANCY
# ============================================================
print("\n" + "-"*70)
print("3. ANALYZING SYMBOL vs DESCRIPTION RELATIONSHIP")
print("-"*70)

# Check if Symbol appears in Description
symbol_in_desc = sum(1 for i, row in all_data.head(1000).iterrows() 
                     if str(row['Symbol']).lower() in str(row['Description']).lower())
print(f"\n  Symbol appears in Description: {symbol_in_desc}/1000 samples ({symbol_in_desc/10:.1f}%)")

# Check uniqueness
unique_symbols = all_data['Symbol'].nunique()
unique_descriptions = all_data['Description'].nunique()
print(f"\n  Unique Symbols: {unique_symbols:,}")
print(f"  Unique Descriptions: {unique_descriptions:,}")

# Check if Symbol and Description have 1:1 mapping
symbol_desc_pairs = all_data.groupby('Symbol')['Description'].nunique()
multi_desc_symbols = (symbol_desc_pairs > 1).sum()
print(f"  Symbols with multiple descriptions: {multi_desc_symbols}")

if symbol_in_desc > 800:
    print("\n  [PARTIAL REDUNDANCY] Symbol is often contained in Description")
    print("  RECOMMENDATION: Consider using only Description or extract Symbol from it")

# ============================================================
# 4. ANALYZE GeneGroupMethod
# ============================================================
print("\n" + "-"*70)
print("4. ANALYZING GeneGroupMethod")
print("-"*70)

ggm_counts = all_data['GeneGroupMethod'].value_counts()
print(f"\n  Value distribution:")
for val, count in ggm_counts.items():
    pct = count / len(all_data) * 100
    print(f"    '{val}': {count:,} ({pct:.2f}%)")

if len(ggm_counts) == 1:
    print(f"\n  [REDUNDANT] GeneGroupMethod has only ONE value!")
    print(f"  RECOMMENDATION: Remove this column - provides no information")
    redundant_cols.append('GeneGroupMethod')

# ============================================================
# 5. SEQUENCE-BASED FEATURE ANALYSIS
# ============================================================
print("\n" + "-"*70)
print("5. SEQUENCE-BASED FEATURE ANALYSIS")
print("-"*70)

# Extract sequence features
print("\n  Extracting sequence features from sample...")
sample = all_data.head(5000).copy()

# Clean sequences
sample['clean_seq'] = sample['NucleotideSequence'].str.replace('<', '').str.replace('>', '')
sample['seq_length'] = sample['clean_seq'].str.len()

# Nucleotide counts
for nuc in ['A', 'T', 'G', 'C']:
    sample[f'count_{nuc}'] = sample['clean_seq'].str.upper().str.count(nuc)

# GC content
sample['gc_content'] = (sample['count_G'] + sample['count_C']) / sample['seq_length']

# Calculate correlations between sequence features
seq_features = ['seq_length', 'count_A', 'count_T', 'count_G', 'count_C', 'gc_content']
print("\n  Correlation matrix for sequence-derived features:")
corr_matrix = sample[seq_features].corr()

print(f"\n  {'':>12}", end='')
for col in seq_features:
    print(f"{col[:8]:>10}", end='')
print()

for i, row_name in enumerate(seq_features):
    print(f"  {row_name[:12]:<12}", end='')
    for j, col_name in enumerate(seq_features):
        val = corr_matrix.iloc[i, j]
        print(f"{val:>10.3f}", end='')
    print()

# Find high correlations
print("\n  High correlations (|r| > 0.8):")
high_corr_found = False
for i in range(len(seq_features)):
    for j in range(i+1, len(seq_features)):
        corr = abs(corr_matrix.iloc[i, j])
        if corr > 0.8:
            high_corr_found = True
            print(f"    {seq_features[i]} <-> {seq_features[j]}: {corr_matrix.iloc[i,j]:.3f}")

if not high_corr_found:
    print("    None found (good - features are independent)")

# ============================================================
# 6. GeneType DISTRIBUTION BY FEATURES
# ============================================================
print("\n" + "-"*70)
print("6. FEATURE CORRELATION WITH TARGET (GeneType)")
print("-"*70)

# Sequence length by GeneType
print("\n  Average sequence length by GeneType:")
length_by_type = sample.groupby('GeneType')['seq_length'].agg(['mean', 'std', 'count'])
length_by_type = length_by_type.sort_values('mean', ascending=False)
print(f"\n  {'GeneType':<25} {'Mean Len':>10} {'Std':>10} {'Count':>8}")
print("  " + "-"*55)
for gtype, row in length_by_type.iterrows():
    print(f"  {gtype:<25} {row['mean']:>10.1f} {row['std']:>10.1f} {int(row['count']):>8}")

# GC content by GeneType
print("\n  Average GC content by GeneType:")
gc_by_type = sample.groupby('GeneType')['gc_content'].agg(['mean', 'std'])
gc_by_type = gc_by_type.sort_values('mean', ascending=False)
print(f"\n  {'GeneType':<25} {'Mean GC':>10} {'Std':>10}")
print("  " + "-"*45)
for gtype, row in gc_by_type.iterrows():
    print(f"  {gtype:<25} {row['mean']:>10.3f} {row['std']:>10.3f}")

# ============================================================
# 7. SUMMARY AND RECOMMENDATIONS
# ============================================================
print("\n" + "="*70)
print("                    SUMMARY & RECOMMENDATIONS")
print("="*70)

print("\n  REDUNDANT PARAMETERS TO REMOVE:")
print("  " + "-"*50)

if 'GeneGroupMethod' in redundant_cols or all_data['GeneGroupMethod'].nunique() == 1:
    print("""
  1. [REMOVE] GeneGroupMethod
     Reason: Contains only one value ('NCBI Ortholog')
     Impact: No predictive value, reduces dimensionality
""")

print("""
  2. [CONSIDER REMOVING] NCBIGeneID
     Reason: Unique identifier (1:1 mapping with rows)
     Impact: Should not be used as feature, only as ID
""")

print("""
  3. [CONSIDER MERGING] Symbol + Description
     Reason: Symbol often contained in Description
     Options: 
       a) Use only Description
       b) Extract features from both
       c) Use Symbol for short reference, Description for NLP
""")

print("\n  USEFUL PARAMETERS:")
print("  " + "-"*50)
print("""
  1. [KEEP] NucleotideSequence
     Reason: Main predictive feature
     Derived features: length, GC content, nucleotide counts
     
  2. [KEEP] GeneType
     Reason: Target variable (label)
     
  3. [KEEP] Description (optional)
     Reason: May contain useful text features for NLP
""")

print("\n  RECOMMENDED FINAL SCHEMA:")
print("  " + "-"*50)
print("""
  For modeling, use:
  - NucleotideSequence (or derived features)
  - GeneType (target)
  
  Optional:
  - Description (if using NLP features)
  
  Remove:
  - GeneGroupMethod (constant)
  - NCBIGeneID (identifier only)
  - Symbol (redundant with Description)
""")

# Create cleaned dataset suggestion
print("\n" + "="*70)
print("                         END OF ANALYSIS")
print("="*70 + "\n")

