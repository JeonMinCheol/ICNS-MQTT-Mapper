import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Seed setting
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Constants
NUM_CLIENTS = 1 #3 #5 #7 #10 #20
NUM_EPOCHS = 2 #200
NUM_ROUNDS = 50 #50, #100 #200
BATCH_SIZE = 32
ATTENTION_DROPOUT = 0.2

# Load dataset
EEG_train_features = pd.read_csv('./home/work/YURI/Federated-Learning/X_EEG_train_fill.csv')
ECG_train_features = pd.read_csv('./home/work/YURI/Federated-Learning/X_ECG_train_fill.csv')
EEG_ECG_train_labels = pd.read_csv('./home/work/YURI/Federated-Learning/y_eeg_ecg_train_fill.csv')

EEG_test_features = pd.read_csv('./home/work/YURI/Federated-Learning/X_EEG_test_fill.csv')
ECG_test_features = pd.read_csv('./home/work/YURI/Federated-Learning/X_ECG_test_fill.csv')
EEG_ECG_test_labels = pd.read_csv('./home/work/YURI/Federated-Learning/y_eeg_ecg_test_fill.csv')

# Prepare data
## train set
X_EEG_train = np.array(EEG_train_features, dtype=np.float32)
X_ECG_train = np.array(ECG_train_features, dtype=np.float32)
y_EEG_ECG_train = np.array(EEG_ECG_train_labels, dtype=np.float32)

## test set
X_EEG_test = np.array(EEG_test_features, dtype=np.float32)
X_ECG_test = np.array(ECG_test_features, dtype=np.float32)
y_EEG_ECG_test = np.array(EEG_ECG_test_labels, dtype=np.float32)

# Split data among clients
client_ids = ['client_{}'.format(i) for i in range(NUM_CLIENTS)]

data_dict_eeg = {client_id: X_EEG_train[i::NUM_CLIENTS] for i, client_id in enumerate(client_ids)}
data_dict_ecg = {client_id: X_ECG_train[i::NUM_CLIENTS] for i, client_id in enumerate(client_ids)}
target_dict_eeg_ecg = {client_id: y_EEG_ECG_train[i::NUM_CLIENTS] for i, client_id in enumerate(client_ids)}

test_data_dict_eeg = {client_id: X_EEG_test[i::NUM_CLIENTS] for i, client_id in enumerate(client_ids)}
test_data_dict_ecg = {client_id: X_ECG_test[i::NUM_CLIENTS] for i, client_id in enumerate(client_ids)}
test_target_dict_eeg_ecg = {client_id: y_EEG_ECG_test[i::NUM_CLIENTS] for i, client_id in enumerate(client_ids)}

# train Example data for input spec
example_data = tf.data.Dataset.from_tensor_slices(((X_EEG_train, X_ECG_train), y_EEG_ECG_train)).batch(BATCH_SIZE)

# test Example data for input spec
test_example_data = tf.data.Dataset.from_tensor_slices(((X_EEG_test, X_ECG_test), y_EEG_ECG_test)).batch(BATCH_SIZE)

# create federated data correctly
def make_federated_data(client_ids):
    return [
        tf.data.Dataset.from_tensor_slices(
            ((data_dict_eeg[client_id], data_dict_ecg[client_id]),
             target_dict_eeg_ecg[client_id]))
        .batch(BATCH_SIZE)
        .repeat(NUM_EPOCHS)
        for client_id in client_ids
    ]


# Define test clients    
train_clients = client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(train_clients)

# Define scaled dot product attention function
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights

class CrossAttention(layers.Layer):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = layers.Dense(heads * dim_head, use_bias=False)
        self.to_k = layers.Dense(heads * dim_head, use_bias=False)
        self.to_v = layers.Dense(heads * dim_head, use_bias=False)
        self.dropout = layers.Dropout(dropout)
        self.to_out = layers.Dense(query_dim)

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = tf.reshape(q, (batch_size, -1, self.heads, self.dim_head))
        k = tf.reshape(k, (batch_size, -1, self.heads, self.dim_head))
        v = tf.reshape(v, (batch_size, -1, self.heads, self.dim_head))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.heads * self.dim_head))

        out = self.to_out(attention_output)
        return out

# Feed Forward Network Layer
class FeedForward(layers.Layer):
    def __init__(self, dff: int, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ])
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        output = self.seq(x)
        output = self.add([x, output])
        output = self.layer_norm(output)
        return output
    
class EncoderLayer(layers.Layer):
    def __init__(self, dff, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = FeedForward(dff, d_model, dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        x = tf.expand_dims(x, axis=1)  
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# creating a model with keras
def create_keras_model():
    with tf.device("/gpu:0"):
        eeg_inputs = tf.keras.Input((X_EEG_train.shape[1],), name="eeg_inputs")
        ecg_inputs = tf.keras.Input((X_ECG_train.shape[1],), name="ecg_inputs")

        # Cross-Attention between EEG and ECG
        eeg_features_expanded = tf.expand_dims(eeg_inputs, axis=1)
        ecg_features_expanded = tf.expand_dims(ecg_inputs, axis=1)

        eeg2ecg_features = CrossAttention(query_dim=128, heads=8, dim_head=16, dropout=0.1)(
            eeg_features_expanded, ecg_features_expanded, ecg_features_expanded)
        ecg2eeg_features = CrossAttention(query_dim=128, heads=8, dim_head=16, dropout=0.1)(
            ecg_features_expanded, eeg_features_expanded, eeg_features_expanded)

        eeg2ecg_features = tf.squeeze(eeg2ecg_features, axis=1)
        ecg2eeg_features = tf.squeeze(ecg2eeg_features, axis=1)

        combined_features = tf.concat([eeg2ecg_features, ecg2eeg_features], axis=-1)
        
        cross_attention_output = EncoderLayer(dff=2048, d_model=256, num_heads=8, dropout_rate=0.1)(combined_features, training=True)
        
        self_attention_output = EncoderLayer(dff=2048, d_model=256, num_heads=8, dropout_rate=0.1)(cross_attention_output, training=True)

        self_attention_output_squeezed = tf.squeeze(self_attention_output, axis=2)
        self_attention_output_flattened = GlobalAveragePooling1D()(self_attention_output_squeezed)

        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(self_attention_output_flattened)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.2)(x)

        outputs = Dense(3, activation='softmax', name='output')(x)
        model = Model(inputs=[eeg_inputs, ecg_inputs], outputs=outputs)
   
    return model


def model_fn():
    with tf.device("/gpu:0"):
        keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=example_data.element_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    
# Custom aggregation function to aggregate all encoder parameters
def aggregate_encoder_parameters(model_weights):
    with tf.device("/gpu:0"):
        keras_model = create_keras_model()
        keras_model.set_weights(model_weights.trainable)
        encoder_weights = [layer.get_weights() for layer in keras_model.layers if isinstance(layer, (CrossAttention, tf.keras.layers.MultiHeadAttention))]
    return encoder_weights

# Update the global model with aggregated encoder parameters
def update_model_with_encoder_weights(model_weights, encoder_weights):
    with tf.device("/gpu:0"):
        keras_model = create_keras_model()
        keras_model.set_weights(model_weights.trainable)
        weight_idx = 0
        for layer in keras_model.layers:
            if isinstance(layer, (CrossAttention, tf.keras.layers.MultiHeadAttention)):
                layer.set_weights(encoder_weights[weight_idx])
                weight_idx += 1
        updated_trainable_weights = keras_model.trainable_weights
        updated_non_trainable_weights = keras_model.non_trainable_weights
        updated_model_weights = tff.learning.models.ModelWeights(
            trainable=updated_trainable_weights,
            non_trainable=updated_non_trainable_weights)
    return updated_model_weights
