# Gene Data Summary Report
# =========================

import pandas as pd
import numpy as np

print("\n" + "="*70)
print("           GENE DATA SUMMARY REPORT")
print("="*70)

# Load data
print("\nLoading datasets...")
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
validation = pd.read_csv('validation.csv', index_col=0)
print("Done!")

# ============================================================
# SECTION 1: BASIC INFORMATION
# ============================================================
print("\n" + "-"*70)
print("1. BASIC DATASET INFORMATION")
print("-"*70)

total = len(train) + len(test) + len(validation)
print(f"\n{'Dataset':<15} {'Rows':>10} {'Columns':>10} {'Percentage':>12}")
print("-"*50)
print(f"{'Train':<15} {len(train):>10,} {len(train.columns):>10} {len(train)/total*100:>11.1f}%")
print(f"{'Test':<15} {len(test):>10,} {len(test.columns):>10} {len(test)/total*100:>11.1f}%")
print(f"{'Validation':<15} {len(validation):>10,} {len(validation.columns):>10} {len(validation)/total*100:>11.1f}%")
print("-"*50)
print(f"{'TOTAL':<15} {total:>10,}")

# ============================================================
# SECTION 2: COLUMN DETAILS (PARAMETERS)
# ============================================================
print("\n" + "-"*70)
print("2. PARAMETERS (COLUMNS) DESCRIPTION")
print("-"*70)

for col in train.columns:
    print(f"\n  [{col}]")
    print(f"    Type: {train[col].dtype}")
    print(f"    Missing values: {train[col].isna().sum()}")
    print(f"    Unique values: {train[col].nunique():,}")
    
    if train[col].dtype == 'object' and train[col].nunique() < 20:
        print(f"    Categories: {list(train[col].unique()[:10])}")

# ============================================================
# SECTION 3: LABEL DISTRIBUTION (GeneType)
# ============================================================
print("\n" + "-"*70)
print("3. LABEL DISTRIBUTION (TARGET: GeneType)")
print("-"*70)

all_data = pd.concat([train, test, validation])
gene_types = all_data['GeneType'].value_counts()

print(f"\nNumber of classes: {len(gene_types)}")
print(f"\n{'GeneType':<25} {'Count':>10} {'Percent':>10}")
print("-"*50)

for gtype, count in gene_types.items():
    pct = count / len(all_data) * 100
    bar = "#" * int(pct / 2)
    print(f"{gtype:<25} {count:>10,} {pct:>9.2f}% {bar}")

# Per dataset breakdown
print("\n\nLabel distribution per dataset:")
print(f"\n{'GeneType':<25} {'Train':>10} {'Test':>10} {'Valid':>10}")
print("-"*60)

for gtype in gene_types.index:
    t = (train['GeneType'] == gtype).sum()
    te = (test['GeneType'] == gtype).sum()
    v = (validation['GeneType'] == gtype).sum()
    print(f"{gtype:<25} {t:>10,} {te:>10,} {v:>10,}")

# ============================================================
# SECTION 4: SEQUENCE ANALYSIS
# ============================================================
print("\n" + "-"*70)
print("4. NUCLEOTIDE SEQUENCE ANALYSIS")
print("-"*70)

for name, df in [("Train", train), ("Test", test), ("Validation", validation)]:
    lengths = df['NucleotideSequence'].str.len()
    print(f"\n  {name}:")
    print(f"    Min length:    {lengths.min():>6}")
    print(f"    Max length:    {lengths.max():>6}")
    print(f"    Mean length:   {lengths.mean():>6.1f}")
    print(f"    Median length: {lengths.median():>6.1f}")

# Length distribution
print("\n  Sequence Length Distribution (Train):")
lengths = train['NucleotideSequence'].str.len()
bins = [0, 100, 200, 300, 500, 1000, float('inf')]
labels = ['0-100', '101-200', '201-300', '301-500', '501-1000', '>1000']
dist = pd.cut(lengths, bins=bins, labels=labels).value_counts().sort_index()

for label, count in dist.items():
    pct = count / len(train) * 100
    bar = "#" * int(pct / 3)
    print(f"    {label:>10}: {count:>6,} ({pct:>5.1f}%) {bar}")

# ============================================================
# SECTION 5: DATA QUALITY
# ============================================================
print("\n" + "-"*70)
print("5. DATA QUALITY CHECK")
print("-"*70)

for name, df in [("Train", train), ("Test", test), ("Validation", validation)]:
    print(f"\n  {name}:")
    print(f"    Missing values:  {df.isna().sum().sum()}")
    print(f"    Duplicate rows:  {df.duplicated().sum()}")
    print(f"    Unique GeneIDs:  {df['NCBIGeneID'].nunique():,}")

# ============================================================
# SECTION 6: SUMMARY
# ============================================================
print("\n" + "-"*70)
print("6. SUMMARY")
print("-"*70)

print(f"""
  Problem Type: Multi-class Classification
  Target Variable: GeneType
  Number of Classes: {len(gene_types)}
  Total Samples: {total:,}
  
  Main Feature: NucleotideSequence (DNA/RNA sequence)
  
  Class Imbalance: YES (PSEUDO dominates with ~{gene_types.iloc[0]/total*100:.0f}%)
  
  Parameters:
    - NCBIGeneID: Unique gene identifier
    - Symbol: Gene symbol/name
    - Description: Full gene description
    - GeneType: Gene classification (TARGET)
    - GeneGroupMethod: Grouping method
    - NucleotideSequence: DNA/RNA sequence
""")

print("="*70)
print("                    END OF REPORT")
print("="*70 + "\n")

