from module import TubeletEmbedding, PositionalEncoder
from model import NN
from dataloader import dataloader
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (64, 128, 128, 3)
NUM_CLASSES = 2

LEARNING_RATE = 1e-4

EPOCHS = 50

PATCH_SIZE = (16, 32, 32)

LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

Fight = np.load('npy_data/Fight.npy')
NonFight = np.load('npy_data/NonFight.npy')
videos = np.concatenate((Fight, NonFight), axis=0)
videos = np.asarray(videos)
labels = np.concatenate([np.ones(len(Fight)), np.zeros(len(NonFight))])
X_train, X_val, y_train, y_val = train_test_split(videos, labels, test_size=0.05, random_state=13822)

trainloader = dataloader(X_train, y_train, "train", BATCH_SIZE)
validloader = dataloader(X_val, y_val, "valid", BATCH_SIZE)

model = NN(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
            )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'model/model_1210_4.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.load_weights('model/model_1210_3.h5')

model.fit(trainloader,validation_data=validloader,epochs=EPOCHS,callbacks=model_checkpoint_callback)