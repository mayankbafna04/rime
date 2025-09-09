import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                           recall_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
from data_processing import preprocess_genbank_data_enhanced
from rime_optimizer import EnhancedRimeOptimizer
from cnn_model import set_dataset_for_fitness, fitness_function, build_flexible_cnn, build_advanced_cnn


def ensemble_predictions(models, X_test):
    """
    Ensemble predictions from multiple models.
    """
    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)
    
    # Average predictions
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions


def train_with_cross_validation(X, y, best_params, n_splits=5):
    """
    Train multiple models using cross-validation for ensemble.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    val_scores = []
    
    print(f"\nTraining {n_splits}-fold cross-validation ensemble...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Training fold {fold}/{n_splits}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build model with best parameters
        # Use advanced CNN for final training
        params_advanced = np.append(best_params, 1)  # Flag for advanced model
        model = build_advanced_cnn(params_advanced)
        
        # Calculate class weights
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Determine batch size
        batch_sizes = [16, 32, 64, 128]
        batch_idx = int(best_params[12]) if len(best_params) > 12 else 1
        batch_size = batch_sizes[batch_idx % len(batch_sizes)]
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate on validation set
        val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_pred)
        val_scores.append(val_acc)
        
        models.append(model)
        print(f"    Fold {fold} validation accuracy: {val_acc:.4f}")
    
    print(f"  Average validation accuracy: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")
    
    return models


def augment_data(X, y, augmentation_factor=2):
    """
    Simple data augmentation for genomic data.
    """
    print(f"Augmenting data by factor of {augmentation_factor}...")
    
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(augmentation_factor - 1):
        # Add noise
        noise = np.random.normal(0, 0.01, X.shape)
        X_noisy = X + noise
        
        # Random rotation (90 degree increments)
        X_rotated = np.rot90(X, k=np.random.randint(1, 4), axes=(1, 2))
        
        X_aug.extend([X_noisy, X_rotated])
        y_aug.extend([y, y])
    
    X_augmented = np.concatenate(X_aug, axis=0)
    y_augmented = np.concatenate(y_aug, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X_augmented))
    
    return X_augmented[indices], y_augmented[indices]


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # --- 1. Enhanced Data Loading and Preprocessing ---
    print("=" * 70)
    print("ENHANCED GENOMIC EXON PREDICTION SYSTEM")
    print("=" * 70)
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str(BASE_DIR / 'data')
    
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    print("\n--- Phase 1: Advanced Feature Extraction ---")
    X, y = preprocess_genbank_data_enhanced(DATA_DIR)
    
    # Data augmentation for better training
    X, y = augment_data(X, y, augmentation_factor=2)
    print(f"After augmentation: {X.shape[0]} samples")
    
    # Split data
    X_train_opt, X_test, y_train_opt, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Further split for validation during optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_opt, y_train_opt, test_size=0.15, random_state=42, stratify=y_train_opt
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Set dataset for fitness function
    set_dataset_for_fitness(X_train, y_train)
    
    # --- 2. Enhanced RIME Optimization ---
    print("\n--- Phase 2: Enhanced RIME Architecture Optimization ---")
    
    # Extended search space for better exploration
    # More conservative search space to avoid edge cases
    search_space = [
        (0, 1), (0, 3),   # Layer 1: type and params  
        (0, 1), (0, 3),   # Layer 2
        (0, 1), (0, 3),   # Layer 3
        (0, 1), (0, 3),   # Layer 4
        (0, 1), (0, 3),   # Layer 5
        (32, 256),        # FC hidden units (increased minimum)
        (0, 3),           # Dropout index
        (0, 2),           # Optimizer index (reduced to avoid issues)
        (0, 3),           # Batch size index
    ]

    
    # Use enhanced RIME optimizer
    rime_optimizer = EnhancedRimeOptimizer(
        objective_func=fitness_function,
        search_space=search_space,
        population_size=25,  # Larger population
        max_iterations=40    # More iterations
    )
    
    best_params, best_fitness = rime_optimizer.optimize()
    print(f"\nOptimization Complete!")
    print(f"Best fitness (1-F1): {best_fitness:.4f}")
    print(f"Best parameters: {best_params}")
    
    # --- 3. Ensemble Training with Cross-Validation ---
    print("\n--- Phase 3: Training Ensemble Models ---")
    
    # Train ensemble of models
    ensemble_models = train_with_cross_validation(X_train_opt, y_train_opt, best_params, n_splits=5)
    
    # --- 4. Final Evaluation on Test Set ---
    print("\n--- Phase 4: Final Evaluation on Hold-Out Test Set ---")
    
    # Get ensemble predictions
    y_pred_prob = ensemble_predictions(ensemble_models, X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Accuracy:  {accuracy:.4f} (Paper RIME-GENSCAN: 0.9739)")
    print(f"  Precision: {precision:.4f} (Paper RIME-GENSCAN: 0.8862)")
    print(f"  Recall:    {recall:.4f} (Paper RIME-GENSCAN: 0.8258)")
    print(f"  F1-Score:  {f1:.4f} (Paper RIME-GENSCAN: 0.8549)")
    print(f"  AUC:       {auc:.4f} (Paper RIME-GENSCAN: 0.9879)")
    print("=" * 70)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Intron', 'Exon']))
    
    # --- 5. Visualization ---
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Intron', 'Exon'], 
                yticklabels=['Intron', 'Exon'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Enhanced Model - Confusion Matrix')
    plt.savefig('enhanced_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Optimization history plot
    if hasattr(rime_optimizer, 'best_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(rime_optimizer.best_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (1 - F1 Score)')
        plt.title('RIME Optimization Convergence')
        plt.grid(True, alpha=0.3)
        plt.savefig('optimization_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Results saved to disk")
    print("=" * 70)


if __name__ == '__main__':
    main()