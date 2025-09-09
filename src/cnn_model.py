import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                    BatchNormalization, ReLU, Input, Dropout,
                                    GlobalAveragePooling2D, Add, Multiply,
                                    LayerNormalization, LeakyReLU, PReLU)
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# Global variables to hold the dataset
X_train_val, y_train_val = None, None

def set_dataset_for_fitness(X, y):
    """Makes the dataset available to the fitness function."""
    global X_train_val, y_train_val
    X_train_val, y_train_val = X, y

def _ensure_param_array(params, min_len=14):
    """Enhanced parameter validation with comprehensive error handling."""
    try:
        # Handle None input
        if params is None:
            return [0] * min_len
            
        # Handle different input types
        if isinstance(params, (int, float, np.integer, np.floating)):
            params = [float(params)]
        elif hasattr(params, '__iter__'):
            # Convert to list and ensure all elements are numeric
            params = []
            for p in params:
                try:
                    val = float(p)
                    params.append(val if np.isfinite(val) else 0.0)
                except (ValueError, TypeError):
                    params.append(0.0)
        else:
            params = [0.0] * min_len
            
        # Ensure minimum length
        while len(params) < min_len:
            params.append(0.0)
            
        return params[:min_len]  # Truncate if too long
        
    except Exception as e:
        print(f"Parameter validation error: {e}")
        return [0.0] * min_len

def build_flexible_cnn(params, input_shape=(16, 16, 1)):
    """
    Fixed wrapper function with proper layer transitions and error handling.
    """
    try:
        # Validate and pad params for safety
        params = _ensure_param_array(params, min_len=14)
        
        # Validate input shape
        if not input_shape or len(input_shape) != 3:
            input_shape = (16, 16, 1)
        
        # During optimization, use simpler model for speed
        if len(params) > 13 and params[13] == 1:
            return build_advanced_cnn(params, input_shape)
        
        # Build simplified model with proper layer management
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        filter_options = [16, 32, 64, 128]
        kernel_options = [3, 5, 7]
        
        # Track whether we're still in conv layers (4D) or have transitioned to dense (2D)
        is_conv_phase = True
        
        # Build layers based on parameters (5 pairs: layer_type, param_idx)
        first_ten = list(params[:10]) + [0] * (10 - len(params[:10]))
        layer_params = np.array(first_ten).reshape(5, 2)
        
        for i in range(5):
            layer_type = int(layer_params[i][0]) % 2  # Ensure 0 or 1
            param_idx = int(layer_params[i][1]) % 4   # Ensure valid index
            
            if layer_type == 0 and is_conv_phase:  # Conv layer
                num_filters = filter_options[param_idx % len(filter_options)]
                kernel_size = kernel_options[param_idx % len(kernel_options)]
                
                model.add(Conv2D(num_filters, kernel_size, padding='same'))
                model.add(BatchNormalization())
                model.add(ReLU())
                
                # Conditional pooling - check current spatial dimensions
                current_height = model.output_shape[1]
                current_width = model.output_shape[2]
                
                if (current_height is not None and current_width is not None and
                    current_height > 4 and current_width > 4 and param_idx % 2 == 0):
                    model.add(MaxPooling2D(2))
                
            else:  # Transition to dense layers
                if is_conv_phase:
                    # First time transitioning from conv to dense
                    model.add(GlobalAveragePooling2D())
                    is_conv_phase = False
                
                # Add dense layer
                units = filter_options[param_idx % len(filter_options)] * 4
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(0.3))
        
        # Ensure we have proper transition to dense if still in conv phase
        if is_conv_phase:
            model.add(GlobalAveragePooling2D())
        
        # Final dense layers
        hidden_units = max(int(params[10]), 16)  # Ensure minimum units
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        # Safe optimizer selection
        optimizer_options = [
            tf.keras.optimizers.Adam(learning_rate=0.001),
            tf.keras.optimizers.RMSprop(learning_rate=0.001),
            tf.keras.optimizers.Nadam(learning_rate=0.001),
        ]
        
        # Try to add AdamW if available
        try:
            optimizer_options.append(tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01))
        except:
            pass
        
        optimizer_idx = int(params[12]) % len(optimizer_options)
        optimizer = optimizer_options[optimizer_idx]
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
        
    except Exception as e:
        print(f"Error in model building: {e}")
        # Return a simple baseline model
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(2),
            Conv2D(64, 3, padding='same', activation='relu'),
            GlobalAveragePooling2D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

def build_advanced_cnn(params, input_shape=(16, 16, 1)):
    """
    Builds an advanced CNN with proper error handling.
    """
    try:
        # Normalize and pad params for safety
        params = _ensure_param_array(params, min_len=14)
        
        inputs = Input(shape=input_shape)
        
        # Parameter interpretation
        filter_options = [16, 32, 64, 128]
        kernel_options = [3, 5, 7]
        dropout_rates = [0.2, 0.3, 0.4, 0.5]
        activation_options = ['relu', 'leaky_relu', 'prelu', 'elu']
        
        # Initial convolution
        x = Conv2D(32, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Dynamic architecture based on parameters
        first_ten = list(params[:10]) + [0] * (10 - len(params[:10]))
        layer_params = np.array(first_ten).reshape(5, 2)
        
        for i in range(5):
            layer_type = int(layer_params[i][0]) % 2
            param_idx = int(layer_params[i][1]) % 4
            
            if layer_type == 0:  # Residual block with attention
                num_filters = filter_options[param_idx % len(filter_options)]
                x = residual_block(x, num_filters, use_attention=True)
            else:  # Standard conv block
                num_filters = filter_options[param_idx % len(filter_options)]
                kernel_size = kernel_options[param_idx % len(kernel_options)]
                
                x = Conv2D(num_filters, kernel_size, padding='same')(x)
                x = BatchNormalization()(x)
                
                # Dynamic activation
                act_idx = param_idx % len(activation_options)
                if activation_options[act_idx] == 'leaky_relu':
                    x = LeakyReLU(0.2)(x)
                elif activation_options[act_idx] == 'prelu':
                    x = PReLU()(x)
                elif activation_options[act_idx] == 'elu':
                    x = tf.keras.layers.ELU()(x)
                else:
                    x = ReLU()(x)
                
                # Adaptive pooling only if spatial dimensions allow
                if (x.shape[1] is not None and x.shape[2] is not None and
                    x.shape[1] > 2 and x.shape[2] > 2 and param_idx % 3 == 0):
                    x = MaxPooling2D(2)(x)
        
        # Global features
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers with advanced regularization
        hidden_units = max(int(params[10]), 16)
        dropout_idx = int(params[11]) % len(dropout_rates)
        dropout_rate = dropout_rates[dropout_idx]
        
        # First dense block
        x = Dense(hidden_units, kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)
        
        # Second dense block (half size)
        x = Dense(max(hidden_units // 2, 8), kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate * 0.7)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Advanced optimizer configuration
        optimizer_idx = int(params[12])
        advanced_opts = [
            tf.keras.optimizers.Adam(learning_rate=0.001),
            tf.keras.optimizers.RMSprop(learning_rate=0.001),
            tf.keras.optimizers.Nadam(learning_rate=0.001)
        ]
        
        try:
            advanced_opts.append(tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01))
        except:
            pass
        
        optimizer = advanced_opts[optimizer_idx % len(advanced_opts)]
        
        # Compile with additional metrics
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
        
    except Exception as e:
        print(f"Error in advanced model building: {e}")
        return build_flexible_cnn(params, input_shape)

def fitness_function(params):
    """
    Enhanced fitness function with comprehensive error handling.
    """
    try:
        # Validate global dataset availability
        if X_train_val is None or y_train_val is None:
            print("Error: Dataset not set for fitness evaluation")
            return 1.0
        
        # Normalize params to ensure safe indexing throughout
        params = _ensure_param_array(params, min_len=14)
        
        # Stratified split for better representation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
        )
        
        # Build model
        model = build_flexible_cnn(params)
        
        # Safe batch size selection
        batch_size_options = [16, 32, 64, 128]
        batch_size_idx = int(params[12]) % len(batch_size_options)
        batch_size = batch_size_options[batch_size_idx]
        
        # Class weight to handle imbalance
        class_weight = {0: 1.0, 1: 1.0}
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) == 2:
            label_to_count = dict(zip(unique.tolist(), counts.tolist()))
            total = counts.sum()
            class_weight = {
                0: total / (2 * max(label_to_count.get(0, 1), 1)),
                1: total / (2 * max(label_to_count.get(1, 1), 1))
            }
        
        # Train with early stopping callback for efficiency
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate fitness (minimizing error rate)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        if cm.shape == (2, 2) and cm.sum() > 0:
            tn, fp, fn, tp = cm.ravel()
            # Weighted error considering both precision and recall
            precision = tp / max(tp + fp, 1e-10)
            recall = tp / max(tp + fn, 1e-10)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            # Use 1 - F1 score as fitness (to minimize)
            fitness = 1 - f1
        else:
            fitness = 1.0
        
        # Clean up
        tf.keras.backend.clear_session()
        del model
        
        return float(fitness) if np.isfinite(fitness) else 1.0
        
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        tf.keras.backend.clear_session()  # Clean up on error
        return 1.0

# Keep the existing channel_attention, spatial_attention, and residual_block functions unchanged
def channel_attention(input_tensor, ratio=8):
    """Channel attention mechanism to focus on important feature channels."""
    channel = input_tensor.shape[-1]
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    reduced_units = max(int(channel) // ratio if channel is not None else 1, 1)
    fc1 = Dense(reduced_units, activation='relu')(avg_pool)
    fc2 = Dense(int(channel) if channel is not None else reduced_units, activation='sigmoid')(fc1)
    return Multiply()([input_tensor, tf.reshape(fc2, [-1, 1, 1, int(channel) if channel is not None else reduced_units])])

def spatial_attention(input_tensor):
    """Spatial attention mechanism to focus on important spatial regions."""
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return Multiply()([input_tensor, attention])

def residual_block(x, filters, kernel_size=3, use_attention=True):
    """Residual block with optional attention mechanisms."""
    conv1 = Conv2D(filters, kernel_size, padding='same')(x)
    bn1 = BatchNormalization()(conv1)
    act1 = ReLU()(bn1)
    conv2 = Conv2D(filters, kernel_size, padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    
    if use_attention:
        bn2 = channel_attention(bn2)
        bn2 = spatial_attention(bn2)
    
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
    
    output = Add()([bn2, shortcut])
    output = ReLU()(output)
    return output
