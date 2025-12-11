# Solutions for Class Imbalance in Gene Classification
# =====================================================

import pandas as pd
import numpy as np
from collections import Counter

print("="*70)
print("        SOLUTIONS FOR CLASS IMBALANCE")
print("="*70)

# Load data
train = pd.read_csv('train.csv', index_col=0)
print(f"\nDataset size: {len(train):,} samples")

# Show current imbalance
print("\n" + "-"*70)
print("CURRENT CLASS DISTRIBUTION")
print("-"*70)
class_counts = train['GeneType'].value_counts()
for cls, count in class_counts.items():
    pct = count / len(train) * 100
    bar = "#" * int(pct)
    print(f"  {cls:<25} {count:>6} ({pct:>5.1f}%) {bar}")

# ============================================================
# SOLUTION 1: CLASS WEIGHTS
# ============================================================
print("\n" + "="*70)
print("SOLUTION 1: CLASS WEIGHTS (Recommended - Easy & Effective)")
print("="*70)

print("""
  Best for: Quick fix, works with most algorithms
  Pros: No data modification, preserves original distribution
  Cons: May not be enough for extreme imbalance
""")

# Calculate balanced weights
from sklearn.utils.class_weight import compute_class_weight

classes = train['GeneType'].unique()
y = train['GeneType']
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))

print("  Computed class weights:")
for cls, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    {cls:<25} weight: {weight:.3f}")

print("""
  CODE EXAMPLE:
  -------------
  # For sklearn models:
  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier(class_weight='balanced')
  
  # Or with custom weights:
  clf = RandomForestClassifier(class_weight=class_weights)
  
  # For XGBoost (binary or use sample_weight):
  import xgboost as xgb
  clf = xgb.XGBClassifier(scale_pos_weight=ratio)
  
  # For neural networks (Keras):
  model.fit(X, y, class_weight=class_weights)
""")

# ============================================================
# SOLUTION 2: SMOTE (Synthetic Minority Oversampling)
# ============================================================
print("\n" + "="*70)
print("SOLUTION 2: SMOTE - Synthetic Minority Oversampling")
print("="*70)

print("""
  Best for: When you need more minority samples
  Pros: Creates synthetic samples, improves minority class learning
  Cons: Can create noise, requires imblearn library
  
  INSTALL: pip install imbalanced-learn
""")

print("""
  CODE EXAMPLE:
  -------------
  from imblearn.over_sampling import SMOTE
  from sklearn.model_selection import train_test_split
  
  # Split first, then apply SMOTE only to training data!
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
  
  # Apply SMOTE
  smote = SMOTE(random_state=42)
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
  
  print(f"Before SMOTE: {Counter(y_train)}")
  print(f"After SMOTE: {Counter(y_train_resampled)}")
  
  # Train on resampled data
  clf.fit(X_train_resampled, y_train_resampled)
  
  # Test on original test data (never resample test data!)
  y_pred = clf.predict(X_test)
""")

# ============================================================
# SOLUTION 3: RANDOM UNDERSAMPLING
# ============================================================
print("\n" + "="*70)
print("SOLUTION 3: RANDOM UNDERSAMPLING")
print("="*70)

print("""
  Best for: When majority class is very large
  Pros: Fast, reduces training time
  Cons: Loses information from majority class
""")

print("""
  CODE EXAMPLE:
  -------------
  from imblearn.under_sampling import RandomUnderSampler
  
  rus = RandomUnderSampler(random_state=42)
  X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
  
  # Or with sampling strategy:
  rus = RandomUnderSampler(
      sampling_strategy={
          'PSEUDO': 5000,  # Reduce to 5000 samples
          'BIOLOGICAL_REGION': 3000,
          # Keep minority classes as is
      }
  )
""")

# ============================================================
# SOLUTION 4: COMBINATION (SMOTE + Undersampling)
# ============================================================
print("\n" + "="*70)
print("SOLUTION 4: SMOTEENN / SMOTETomek (Combination)")
print("="*70)

print("""
  Best for: Balanced approach
  Pros: Combines oversampling and cleaning
  Cons: More complex, slower
""")

print("""
  CODE EXAMPLE:
  -------------
  from imblearn.combine import SMOTEENN, SMOTETomek
  
  # SMOTE + ENN (removes noisy samples)
  smote_enn = SMOTEENN(random_state=42)
  X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
  
  # Or SMOTE + Tomek links
  smote_tomek = SMOTETomek(random_state=42)
  X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
""")

# ============================================================
# SOLUTION 5: THRESHOLD ADJUSTMENT
# ============================================================
print("\n" + "="*70)
print("SOLUTION 5: THRESHOLD ADJUSTMENT")
print("="*70)

print("""
  Best for: Fine-tuning predictions after training
  Pros: No retraining needed, can optimize per class
  Cons: Requires probability outputs
""")

print("""
  CODE EXAMPLE:
  -------------
  # Get probability predictions
  y_proba = clf.predict_proba(X_test)
  
  # Default threshold is 0.5, but for imbalanced data:
  # Lower threshold for minority classes
  thresholds = {
      'PSEUDO': 0.5,
      'BIOLOGICAL_REGION': 0.4,
      'PROTEIN_CODING': 0.3,
      'ncRNA': 0.3,
      'snoRNA': 0.2,
      'tRNA': 0.2,
      'OTHER': 0.2,
      'rRNA': 0.15  # Lowest threshold for rarest class
  }
  
  # Custom prediction with thresholds
  def predict_with_threshold(proba, classes, thresholds):
      adjusted = proba.copy()
      for i, cls in enumerate(classes):
          adjusted[:, i] = proba[:, i] / thresholds.get(cls, 0.5)
      return classes[adjusted.argmax(axis=1)]
""")

# ============================================================
# SOLUTION 6: ENSEMBLE WITH BALANCED SUBSETS
# ============================================================
print("\n" + "="*70)
print("SOLUTION 6: BALANCED BAGGING / EASY ENSEMBLE")
print("="*70)

print("""
  Best for: Robust handling of imbalance
  Pros: Uses all data, reduces variance
  Cons: More complex, longer training
""")

print("""
  CODE EXAMPLE:
  -------------
  from imblearn.ensemble import BalancedBaggingClassifier
  from imblearn.ensemble import EasyEnsembleClassifier
  from sklearn.tree import DecisionTreeClassifier
  
  # Balanced Bagging
  bbc = BalancedBaggingClassifier(
      estimator=DecisionTreeClassifier(),
      n_estimators=50,
      random_state=42
  )
  bbc.fit(X_train, y_train)
  
  # Easy Ensemble (multiple AdaBoost classifiers)
  eec = EasyEnsembleClassifier(n_estimators=10, random_state=42)
  eec.fit(X_train, y_train)
""")

# ============================================================
# SOLUTION 7: FOCAL LOSS (For Deep Learning)
# ============================================================
print("\n" + "="*70)
print("SOLUTION 7: FOCAL LOSS (Deep Learning)")
print("="*70)

print("""
  Best for: Neural networks
  Pros: Down-weights easy examples, focuses on hard ones
  Cons: Requires custom loss function
""")

print("""
  CODE EXAMPLE (TensorFlow/Keras):
  --------------------------------
  import tensorflow as tf
  
  def focal_loss(gamma=2.0, alpha=0.25):
      def focal_loss_fn(y_true, y_pred):
          epsilon = tf.keras.backend.epsilon()
          y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
          
          cross_entropy = -y_true * tf.math.log(y_pred)
          weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
          
          return tf.reduce_sum(weight * cross_entropy, axis=-1)
      return focal_loss_fn
  
  model.compile(
      optimizer='adam',
      loss=focal_loss(gamma=2.0),
      metrics=['accuracy']
  )
""")

# ============================================================
# COMPLETE WORKING EXAMPLE
# ============================================================
print("\n" + "="*70)
print("COMPLETE WORKING EXAMPLE")
print("="*70)

print("""
  # Full pipeline for this gene classification dataset:
  
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.metrics import classification_report, f1_score
  from imblearn.over_sampling import SMOTE
  from collections import Counter
  
  # 1. Load data
  train = pd.read_csv('train.csv', index_col=0)
  
  # 2. Clean sequences
  train['seq'] = train['NucleotideSequence'].str.replace('[<>]', '', regex=True)
  
  # 3. Extract k-mer features
  vectorizer = CountVectorizer(analyzer='char', ngram_range=(4,4))
  X = vectorizer.fit_transform(train['seq'])
  y = train['GeneType']
  
  # 4. Split data (BEFORE resampling!)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  
  print("Before SMOTE:", Counter(y_train))
  
  # 5. Apply SMOTE to training data only
  smote = SMOTE(random_state=42)
  X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
  
  print("After SMOTE:", Counter(y_train_res))
  
  # 6. Train with class weights (extra protection)
  clf = RandomForestClassifier(
      n_estimators=100,
      class_weight='balanced',
      random_state=42,
      n_jobs=-1
  )
  clf.fit(X_train_res, y_train_res)
  
  # 7. Evaluate on original test set
  y_pred = clf.predict(X_test)
  
  # 8. Use proper metrics (NOT accuracy!)
  print("\\nClassification Report:")
  print(classification_report(y_test, y_pred))
  print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
  print(f"Weighted F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
""")

# ============================================================
# EVALUATION METRICS FOR IMBALANCED DATA
# ============================================================
print("\n" + "="*70)
print("IMPORTANT: USE CORRECT METRICS!")
print("="*70)

print("""
  DON'T use Accuracy - it's misleading with imbalanced data!
  
  If 60% of data is PSEUDO, a model predicting only PSEUDO 
  gets 60% accuracy but is useless!
  
  USE THESE INSTEAD:
  ------------------
  1. F1 Score (macro) - Equal weight to all classes
  2. F1 Score (weighted) - Weight by class frequency
  3. Precision & Recall per class
  4. Confusion Matrix
  5. ROC-AUC (one-vs-rest for multiclass)
  
  CODE:
  -----
  from sklearn.metrics import (
      classification_report,
      confusion_matrix,
      f1_score,
      precision_recall_fscore_support
  )
  
  # Full report
  print(classification_report(y_test, y_pred))
  
  # Macro F1 (treats all classes equally)
  f1_macro = f1_score(y_test, y_pred, average='macro')
  
  # Confusion matrix
  cm = confusion_matrix(y_test, y_pred)
""")

print("\n" + "="*70)
print("                    SUMMARY")
print("="*70)

print("""
  QUICK START RECOMMENDATION:
  ---------------------------
  1. Use class_weight='balanced' (easiest)
  2. Add SMOTE if still poor minority performance
  3. Always evaluate with F1-macro, not accuracy
  
  For your dataset specifically:
  - PSEUDO (60%) dominates - use undersampling or weights
  - rRNA (<1%) is very rare - may need aggressive SMOTE
  - Consider grouping rare classes if performance is poor
""")

print("\n" + "="*70)
print("                    END")
print("="*70 + "\n")

