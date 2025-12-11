# Classical ML Problems Detection & Solutions
# ============================================

import pandas as pd
import numpy as np
from collections import Counter

print("\n" + "="*70)
print("   CLASSICAL ML PROBLEMS DETECTION & PROPOSED SOLUTIONS")
print("="*70)

# Load data
print("\nLoading data...")
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
validation = pd.read_csv('validation.csv', index_col=0)
all_data = pd.concat([train, test, validation], ignore_index=True)
print(f"Total samples loaded: {len(all_data):,}")

problems_found = []

# ============================================================
# PROBLEM 1: CLASS IMBALANCE
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 1: CLASS IMBALANCE")
print("-"*70)

class_counts = train['GeneType'].value_counts()
total = len(train)
majority_class = class_counts.index[0]
majority_pct = class_counts.iloc[0] / total * 100
minority_class = class_counts.index[-1]
minority_pct = class_counts.iloc[-1] / total * 100
imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[-1]

print(f"\n  Class Distribution (Train):")
print(f"  {'Class':<25} {'Count':>10} {'Percent':>10} {'Ratio':>10}")
print("  " + "-"*55)
for cls, count in class_counts.items():
    ratio = class_counts.iloc[0] / count
    print(f"  {cls:<25} {count:>10,} {count/total*100:>9.1f}% {ratio:>9.1f}x")

print(f"\n  [!] SEVERITY: {'HIGH' if imbalance_ratio > 10 else 'MEDIUM' if imbalance_ratio > 5 else 'LOW'}")
print(f"  [!] Imbalance ratio: {imbalance_ratio:.1f}:1 ({majority_class} vs {minority_class})")

if imbalance_ratio > 5:
    problems_found.append(("Class Imbalance", "HIGH"))

print("""
  SOLUTIONS:
  ----------
  1. RESAMPLING TECHNIQUES:
     - SMOTE (Synthetic Minority Oversampling)
     - Random Undersampling of majority class
     - ADASYN (Adaptive Synthetic Sampling)
     
  2. CLASS WEIGHTS:
     - Use class_weight='balanced' in sklearn
     - Custom weights inversely proportional to frequency
     
  3. ALGORITHM SELECTION:
     - Use algorithms robust to imbalance (XGBoost, LightGBM)
     - Set scale_pos_weight parameter
     
  4. EVALUATION METRICS:
     - Use F1-score, Precision, Recall (not just Accuracy)
     - Use macro/weighted averaging
     - Confusion matrix analysis
     
  5. THRESHOLD TUNING:
     - Adjust classification threshold per class
""")

# ============================================================
# PROBLEM 2: VARIABLE LENGTH SEQUENCES
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 2: VARIABLE LENGTH SEQUENCES")
print("-"*70)

seq_lengths = train['NucleotideSequence'].str.len()
print(f"\n  Sequence Length Statistics:")
print(f"  - Minimum: {seq_lengths.min()}")
print(f"  - Maximum: {seq_lengths.max()}")
print(f"  - Mean: {seq_lengths.mean():.1f}")
print(f"  - Std Dev: {seq_lengths.std():.1f}")
print(f"  - Coefficient of Variation: {seq_lengths.std()/seq_lengths.mean()*100:.1f}%")

length_range = seq_lengths.max() - seq_lengths.min()
cv = seq_lengths.std() / seq_lengths.mean()

if cv > 0.5:
    problems_found.append(("Variable Length Sequences", "MEDIUM"))
    print(f"\n  [!] SEVERITY: MEDIUM (high variance in lengths)")

print("""
  SOLUTIONS:
  ----------
  1. PADDING/TRUNCATION:
     - Pad short sequences to max_length
     - Truncate long sequences
     - Use attention masks for padded positions
     
  2. FIXED-LENGTH FEATURES:
     - Extract k-mer frequencies (e.g., 3-mers, 4-mers)
     - Calculate summary statistics (GC content, length)
     - Use sequence motif counts
     
  3. SEQUENCE MODELS:
     - Use RNN/LSTM (handles variable length)
     - Use 1D CNN with global pooling
     - Transformer with positional encoding
     
  4. BINNING:
     - Group sequences by length ranges
     - Train separate models per bin
""")

# ============================================================
# PROBLEM 3: HIGH DIMENSIONALITY (SEQUENCE DATA)
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 3: HIGH DIMENSIONALITY")
print("-"*70)

max_len = seq_lengths.max()
if max_len > 500:
    problems_found.append(("High Dimensionality", "MEDIUM"))

print(f"\n  If using one-hot encoding:")
print(f"  - Max sequence length: {max_len}")
print(f"  - Nucleotides: 4 (A, T, G, C)")
print(f"  - Potential dimensions: {max_len} x 4 = {max_len * 4:,}")

print(f"\n  If using k-mer features:")
print(f"  - 3-mers: 4^3 = 64 features")
print(f"  - 4-mers: 4^4 = 256 features")
print(f"  - 5-mers: 4^5 = 1,024 features")
print(f"  - 6-mers: 4^6 = 4,096 features")

print(f"\n  [!] SEVERITY: MEDIUM")

print("""
  SOLUTIONS:
  ----------
  1. DIMENSIONALITY REDUCTION:
     - PCA on k-mer frequencies
     - t-SNE/UMAP for visualization
     - Autoencoders for learned representations
     
  2. FEATURE SELECTION:
     - Chi-square test for k-mer importance
     - Mutual information
     - L1 regularization (Lasso)
     
  3. EMBEDDING APPROACHES:
     - Word2Vec for k-mers (DNA2Vec)
     - Pre-trained DNA embeddings
     - Learn embeddings during training
     
  4. EFFICIENT ARCHITECTURES:
     - Use convolutional layers (parameter sharing)
     - Attention mechanisms (focus on important regions)
""")

# ============================================================
# PROBLEM 4: CATEGORICAL TO NUMERICAL CONVERSION
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 4: SEQUENCE ENCODING CHALLENGE")
print("-"*70)

print("""
  Raw sequences are strings that need numerical encoding.
  
  Current format: '<ATGCGATCGATCG...>'
  
  [!] SEVERITY: MEDIUM (standard challenge for sequence data)
""")

problems_found.append(("Sequence Encoding", "MEDIUM"))

print("""
  SOLUTIONS:
  ----------
  1. ONE-HOT ENCODING:
     A = [1,0,0,0], T = [0,1,0,0], G = [0,0,1,0], C = [0,0,0,1]
     
  2. INTEGER ENCODING:
     A = 0, T = 1, G = 2, C = 3
     (Use with embedding layer)
     
  3. K-MER FREQUENCY:
     Count occurrences of all k-length subsequences
     Example (3-mers): AAA, AAT, AAG, AAC, ATA, ...
     
  4. PHYSICOCHEMICAL PROPERTIES:
     - Purine/Pyrimidine encoding
     - Strong/Weak hydrogen bonding
     - Amino/Keto classification
     
  5. ADVANCED ENCODINGS:
     - Chaos Game Representation (CGR)
     - Z-curve encoding
     - DNA2Vec embeddings
""")

# ============================================================
# PROBLEM 5: POTENTIAL DATA LEAKAGE
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 5: POTENTIAL DATA LEAKAGE")
print("-"*70)

# Check for overlap between train/test
train_ids = set(train['NCBIGeneID'].values)
test_ids = set(test['NCBIGeneID'].values)
val_ids = set(validation['NCBIGeneID'].values)

overlap_train_test = len(train_ids & test_ids)
overlap_train_val = len(train_ids & val_ids)
overlap_test_val = len(test_ids & val_ids)

print(f"\n  Checking for ID overlaps:")
print(f"  - Train-Test overlap: {overlap_train_test}")
print(f"  - Train-Validation overlap: {overlap_train_val}")
print(f"  - Test-Validation overlap: {overlap_test_val}")

if overlap_train_test > 0 or overlap_train_val > 0:
    problems_found.append(("Data Leakage", "HIGH"))
    print(f"\n  [!] SEVERITY: HIGH - Overlapping samples detected!")
else:
    print(f"\n  [OK] No overlapping gene IDs found between sets")

# Check Description for potential leakage
desc_has_type = sum(1 for desc, gtype in zip(train['Description'].head(100), train['GeneType'].head(100))
                    if gtype.lower() in desc.lower())
print(f"\n  GeneType appears in Description: {desc_has_type}/100 samples")
if desc_has_type > 50:
    print("  [!] WARNING: Target may leak through Description!")
    problems_found.append(("Label Leakage in Description", "MEDIUM"))

print("""
  SOLUTIONS:
  ----------
  1. PROPER DATA SPLITTING:
     - Ensure no sample overlap between sets
     - Use stratified splitting by GeneType
     
  2. FEATURE AUDIT:
     - Don't use Description if it contains GeneType
     - Remove any features derived from target
     
  3. CROSS-VALIDATION:
     - Use proper CV with no leakage
     - GroupKFold if there are related samples
""")

# ============================================================
# PROBLEM 6: OVERFITTING RISK
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 6: OVERFITTING RISK")
print("-"*70)

n_samples = len(train)
n_minority = class_counts.iloc[-1]
estimated_features = 256  # assuming 4-mer features

print(f"\n  Dataset characteristics:")
print(f"  - Training samples: {n_samples:,}")
print(f"  - Minority class samples: {n_minority:,}")
print(f"  - Estimated features (4-mers): {estimated_features}")
print(f"  - Sample-to-feature ratio: {n_samples/estimated_features:.1f}")
print(f"  - Minority sample-to-feature ratio: {n_minority/estimated_features:.1f}")

if n_minority / estimated_features < 10:
    problems_found.append(("Overfitting Risk", "MEDIUM"))
    print(f"\n  [!] SEVERITY: MEDIUM (minority classes may overfit)")

print("""
  SOLUTIONS:
  ----------
  1. REGULARIZATION:
     - L1/L2 regularization
     - Dropout (for neural networks)
     - Early stopping
     
  2. CROSS-VALIDATION:
     - K-fold CV (k=5 or 10)
     - Stratified CV for imbalanced data
     
  3. SIMPLER MODELS:
     - Start with simple models (Logistic Regression)
     - Gradually increase complexity
     
  4. DATA AUGMENTATION:
     - Reverse complement of sequences
     - Random mutations
     - Subsequence sampling
     
  5. ENSEMBLE METHODS:
     - Bagging to reduce variance
     - Random Forest
""")

# ============================================================
# PROBLEM 7: COMPUTATIONAL COMPLEXITY
# ============================================================
print("\n" + "-"*70)
print("PROBLEM 7: COMPUTATIONAL COMPLEXITY")
print("-"*70)

total_chars = train['NucleotideSequence'].str.len().sum()
print(f"\n  Data size:")
print(f"  - Total nucleotides in train: {total_chars:,}")
print(f"  - Average sequence length: {total_chars/len(train):.0f}")
print(f"  - Estimated memory for one-hot (float32): {total_chars * 4 * 4 / 1e9:.2f} GB")

if total_chars > 10_000_000:
    problems_found.append(("Computational Complexity", "MEDIUM"))
    print(f"\n  [!] SEVERITY: MEDIUM")

print("""
  SOLUTIONS:
  ----------
  1. EFFICIENT DATA LOADING:
     - Use data generators/iterators
     - Batch processing
     - Memory mapping for large files
     
  2. FEATURE COMPRESSION:
     - Use k-mer frequencies instead of full sequences
     - Sparse matrices for one-hot encoding
     
  3. MODEL OPTIMIZATION:
     - Use GPU acceleration
     - Mixed precision training
     - Gradient checkpointing
     
  4. DISTRIBUTED COMPUTING:
     - Multi-GPU training
     - Distributed data processing
""")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("                    PROBLEMS SUMMARY")
print("="*70)

print(f"\n  {'Problem':<40} {'Severity':>10}")
print("  " + "-"*52)
for problem, severity in problems_found:
    emoji = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
    print(f"  {emoji} {problem:<38} {severity:>10}")

print(f"\n  Total problems identified: {len(problems_found)}")

print("\n" + "="*70)
print("                RECOMMENDED ACTION PLAN")
print("="*70)

print("""
  PRIORITY 1 (Before modeling):
  -----------------------------
  [x] Remove GeneGroupMethod column (constant)
  [x] Handle class imbalance (use class weights)
  [x] Encode sequences (k-mer or one-hot)
  [x] Verify no data leakage
  
  PRIORITY 2 (During modeling):
  -----------------------------
  [ ] Use appropriate metrics (F1, not accuracy)
  [ ] Apply regularization
  [ ] Use cross-validation
  [ ] Start with simple models
  
  PRIORITY 3 (Optimization):
  --------------------------
  [ ] Try SMOTE for minority classes
  [ ] Experiment with different k-mer sizes
  [ ] Consider deep learning for complex patterns
  [ ] Ensemble multiple models

  SUGGESTED BASELINE MODEL:
  -------------------------
  1. Extract 4-mer frequencies from sequences
  2. Use Random Forest with class_weight='balanced'
  3. Evaluate with macro F1-score
  4. Use 5-fold stratified cross-validation
""")

print("\n" + "="*70)
print("                    END OF ANALYSIS")
print("="*70 + "\n")

