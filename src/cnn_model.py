import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# Global variables to hold the dataset
X_train_val, y_train_val = None, None

def set_dataset_for_fitness(X, y):
    """Makes the dataset available to the fitness function."""
    global X_train_val, y_train_val
    X_train_val, y_train_val = X, y

def build_flexible_cnn(params, input_shape=(16, 16, 1)):
    """
    Builds a CNN model from a flexible parameter vector optimized by RIME.
    This is an enhancement over the paper's rigid two-stage approach.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Architecture defined by first 5 layers (type and params)
    # Layer Type: 0=Conv, 1=Pool
    layer_params = np.array_split(params[:10], 5) # 5 layers, 2 params each
    filter_options = [8, 16, 32, 64]
    kernel_options = [3, 5, 7]
    pool_options = [2, 3]

    for p in layer_params:
        layer_type = p[0]
        param_idx = p[1]
        if layer_type == 0: # Convolutional Layer
            num_filters = filter_options[param_idx % len(filter_options)]
            kernel_size = kernel_options[param_idx % len(kernel_options)]
            model.add(Conv2D(num_filters, (kernel_size, kernel_size), padding='same'))
            model.add(BatchNormalization())
            model.add(ReLU())
        elif layer_type == 1: # Pooling Layer
            pool_size = pool_options[param_idx % len(pool_options)]
            # Prevent pooling if image is too small
            if model.output_shape[1] > pool_size and model.output_shape[2] > pool_size:
                model.add(MaxPooling2D((pool_size, pool_size)))

    model.add(Flatten())
    
    # FC Layer, Optimizer, and Batch Size
    hidden_units = params[10]
    optimizer_idx = params[11]
    
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # Binary classification

    optimizers = ['adam', 'rmsprop', 'sgd']
    optimizer = optimizers[optimizer_idx]
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def fitness_function(params):
    """
    The objective function for RIME: trains a CNN and returns its Bit Error Rate (BER).
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val)

    try:
        model = build_flexible_cnn(params)
        batch_size_options = [16, 32, 64, 128]
        batch_size = batch_size_options[params[12]]

        model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=0)
        
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Handle cases with only one class in prediction
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        ber = (fp + fn) / (tp + tn + fp + fn)
        return ber if np.isfinite(ber) else 1.0

    except Exception:
        return 1.0 # Penalize failing architectures