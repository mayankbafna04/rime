import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from data_processing import preprocess_genbank_data
from rime_optimizer import RimeOptimizer
from cnn_model import set_dataset_for_fitness, fitness_function, build_flexible_cnn


def main():
    # --- 1. Data Loading and Preprocessing ---
    # Resolve absolute path to the `data` directory relative to repo root
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str(BASE_DIR / 'data')
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    X, y = preprocess_genbank_data(DATA_DIR)
    
    X_train_opt, X_test, y_train_opt, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    set_dataset_for_fitness(X_train_opt, y_train_opt)

    # --- 2. A Single, Powerful Optimization Phase ---
    print("\n--- Starting Unified Architecture & Hyperparameter Optimization ---")
    
    # [Layer1_type, L1_param, L2_type, L2_param, ..., L5_type, L5_param, FC_units, Optimizer, Batch_size]
    # Layer type: 0=Conv, 1=Pool. Param_idx maps to filters/kernels or pool_size.
    search_space = [
        (0, 1), (0, 3), # Layer 1
        (0, 1), (0, 3), # Layer 2
        (0, 1), (0, 3), # Layer 3
        (0, 1), (0, 3), # Layer 4
        (0, 1), (0, 3), # Layer 5
        (32, 512),      # FC hidden units
        (0, 2),         # Optimizer index
        (0, 3)          # Batch size index
    ]
    
    rime_optimizer = RimeOptimizer(
        objective_func=fitness_function,
        search_space=search_space,
        population_size=20, # Increased for better exploration
        max_iterations=30   # Increased for deeper search
    )
    best_params, _ = rime_optimizer.optimize()
    print(f"\nOptimization Complete. Best Parameter Vector Found:\n{best_params}")

    # --- 3. Final Model Training and Evaluation ---
    print("\n--- Training Final Optimized Model on Full Training Data ---")
    final_model = build_flexible_cnn(best_params)
    final_model.summary()
    
    batch_sizes = [16, 32, 64, 128]
    final_batch_size = batch_sizes[best_params[-1]]
    
    # Use early stopping for robust final training
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = final_model.fit(
        X_train_opt, y_train_opt,
        epochs=100,
        batch_size=final_batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- 4. Performance Evaluation on Test Set ---
    print("\n--- Evaluating on Hold-Out Test Set ---")
    y_pred_prob = final_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\n--- Performance Metrics ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f} (Paper RIME-GENSCAN: 0.9739) [cite: 871]")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f} (Paper RIME-GENSCAN: 0.8862) [cite: 871]")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f} (Paper RIME-GENSCAN: 0.8258) [cite: 871]")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f} (Paper RIME-GENSCAN: 0.8549) [cite: 871]")
    print(f"  AUC:       {roc_auc_score(y_test, y_pred_prob):.4f} (Paper RIME-GENSCAN: 0.9879) [cite: 871]")

    # --- 5. Visualization ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Intron', 'Exon'], yticklabels=['Intron', 'Exon'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix of Final Model')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    main()