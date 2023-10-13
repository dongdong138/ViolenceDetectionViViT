import numpy as np
import tensorflow as tf

BATCH_SIZE = 32

@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],
        tf.float32,
    )

    label = tf.cast(label, tf.float32)
    return frames, label

def dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataloader