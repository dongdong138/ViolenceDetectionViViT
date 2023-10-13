import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (64, 128, 128, 3)
NUM_LAYERS = 8
NUM_HEADS = 8
PROJECTION_DIM = 128
LAYER_NORM_EPS = 1e-6
NUM_CLASSES = 2

def NN(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    inputs = layers.Input(shape=input_shape)
 
    patches = tubelet_embedder(inputs)

    encoded_patches = positional_encoder(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
