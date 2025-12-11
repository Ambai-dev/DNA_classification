"""
Gene Classification - Main Training Script
==========================================

This script orchestrates the full ML pipeline:
1. Load and preprocess data
2. Extract features (basic + k-mers)
3. Train multiple models with cross-validation
4. Compare models and evaluate on validation set
5. Save best model and generate reports

Usage:
    python train_model.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add ml_pipeline to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.data_loader import GeneDataLoader
from ml_pipeline.feature_engineering import FeatureExtractor
from ml_pipeline.models import ModelFactory
from ml_pipeline.evaluation import ModelEvaluator
from ml_pipeline.utils import save_model, print_header, print_section, ensure_dir, format_duration


def main():
    """Main training pipeline."""
    
    start_time = time.time()
    
    print_header("GENE CLASSIFICATION - ML TRAINING PIPELINE")
    print("""
    This pipeline will:
    1. Load and preprocess DNA sequence data
    2. Extract features (length, GC content, k-mers)
    3. Train multiple models with balanced class weights
    4. Evaluate using F1-Macro (appropriate for imbalanced data)
    5. Save the best model
    """)
    
    # Ensure results directory exists
    results_dir = 'results'
    ensure_dir(results_dir)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_section("STEP 1: Loading Data")
    
    loader = GeneDataLoader(data_dir='.')
    train_df, val_df, test_df = loader.load_data()
    
    # Preprocess
    train_df = loader.preprocess(train_df, fit_labels=True)
    val_df = loader.preprocess(val_df, fit_labels=False)
    test_df = loader.preprocess(test_df, fit_labels=False)
    
    # Get label mapping
    label_mapping = loader.get_label_mapping()
    print(f"\nLabel mapping: {label_mapping}")
    
    # Print class distribution
    loader.print_class_distribution(train_df, "Training Set")
    
    # Get targets
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print_section("STEP 2: Feature Engineering")
    
    # Initialize feature extractor (4-mers = 256 features + 8 basic)
    extractor = FeatureExtractor(k=4, use_scaler=True)
    
    # Extract features
    print("\nExtracting training features...")
    X_train = extractor.fit_transform(train_df['seq'])
    
    print("\nExtracting validation features...")
    X_val = extractor.transform(val_df['seq'])
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Basic features: 8 (length, gc_content, nucleotide freqs, skews)")
    print(f"K-mer features: {X_train.shape[1] - 8} ({extractor.k}-mers)")
    
    # =========================================================================
    # STEP 3: Initialize Models
    # =========================================================================
    print_section("STEP 3: Initializing Models")
    
    factory = ModelFactory(random_state=42)
    models = factory.get_all_models()
    
    print(f"\nModels to train:")
    for name in models.keys():
        print(f"  • {name}")
    
    # =========================================================================
    # STEP 4: Cross-Validation
    # =========================================================================
    print_section("STEP 4: Cross-Validation (5-Fold Stratified)")
    
    evaluator = ModelEvaluator(label_mapping=label_mapping)
    cv_results = []
    
    for name, model in models.items():
        result = evaluator.cross_validate(model, X_train, y_train, n_splits=5, model_name=name)
        cv_results.append(result)
    
    # =========================================================================
    # STEP 5: Compare Models
    # =========================================================================
    print_section("STEP 5: Model Comparison")
    
    comparison_path = os.path.join(results_dir, 'model_comparison.png')
    comparison_df = evaluator.compare_models(cv_results, save_path=comparison_path)
    
    # Find best model
    best_idx = comparison_df['f1_macro_mean'].idxmax()
    best_model_name = comparison_df.loc[best_idx, 'model']
    best_score = comparison_df.loc[best_idx, 'f1_macro_mean']
    
    # =========================================================================
    # STEP 6: Train Best Model on Full Training Set
    # =========================================================================
    print_section("STEP 6: Training Best Model")
    
    print(f"\nTraining {best_model_name} on full training set...")
    
    # Get fresh instance of best model
    best_model_key = best_model_name.lower().replace(' ', '_')
    best_model = factory.get_model(best_model_key)
    
    # Train
    train_start = time.time()
    best_model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"Training completed in {format_duration(train_time)}")
    
    # =========================================================================
    # STEP 7: Evaluate on Validation Set
    # =========================================================================
    print_section("STEP 7: Validation Set Evaluation")
    
    # Predict on validation
    y_val_pred = best_model.predict(X_val)
    
    # Evaluate
    val_results = evaluator.evaluate(y_val, y_val_pred, f"{best_model_name} (Validation)")
    evaluator.print_classification_report(y_val, y_val_pred)
    
    # Plot confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(y_val, y_val_pred, best_model_name, save_path=cm_path)
    
    # =========================================================================
    # STEP 8: Feature Importance (if available)
    # =========================================================================
    if hasattr(best_model, 'feature_importances_'):
        print_section("STEP 8: Feature Importance Analysis")
        
        importances = best_model.feature_importances_
        feature_names = extractor.get_feature_names()
        
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        
        print("\nTop 20 Most Important Features:")
        print("-" * 50)
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:<20} {importances[idx]:.4f}")
        
        # Save feature importance plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_indices = indices[:20]
        top_names = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{best_model_name} - Top 20 Feature Importances')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        importance_path = os.path.join(results_dir, 'feature_importance.png')
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {importance_path}")
        plt.close()
    
    # =========================================================================
    # STEP 9: Save Best Model
    # =========================================================================
    print_section("STEP 9: Saving Model")
    
    model_path = os.path.join(results_dir, 'best_model.joblib')
    metadata = {
        'model_name': best_model_name,
        'cv_f1_macro': best_score,
        'val_f1_macro': val_results['f1_macro'],
        'n_features': X_train.shape[1],
        'n_classes': len(label_mapping),
        'label_mapping': label_mapping
    }
    save_model(best_model, model_path, metadata)
    
    # Save feature extractor
    import joblib
    extractor_path = os.path.join(results_dir, 'feature_extractor.joblib')
    joblib.dump(extractor, extractor_path)
    print(f"Feature extractor saved to: {extractor_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print_header("TRAINING COMPLETE")
    print(f"""
    Summary:
    --------
    Best Model:           {best_model_name}
    CV F1-Macro:          {best_score:.4f}
    Validation F1-Macro:  {val_results['f1_macro']:.4f}
    Validation Accuracy:  {val_results['accuracy']:.4f}
    
    Total Training Time:  {format_duration(total_time)}
    
    Saved Files:
    ------------
    • {model_path}
    • {extractor_path}
    • {comparison_path}
    • {cm_path}
    """)
    
    if hasattr(best_model, 'feature_importances_'):
        print(f"    • {importance_path}")
    
    print("\n" + "="*70)
    print(" Next Steps:")
    print("="*70)
    print("""
    1. Review the confusion matrix to see where the model struggles
    2. Check feature importances to understand what drives predictions
    3. Consider trying SMOTE oversampling for minority classes
    4. Run predictions on test set when ready for submission
    """)
    
    return best_model, extractor, val_results


if __name__ == "__main__":
    model, extractor, results = main()

