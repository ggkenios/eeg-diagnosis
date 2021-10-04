import tensorflow as tf
from tensorflow import keras

from common.constants import UNITS, RESHAPED, INPUT_DIM, OUTPUT_SIZE, LEARNING_RATE


def model_build():
    """Creating an RNN model"""

    # Model
    rnn_model = keras.models.Sequential(
        [
            keras.layers.LSTM(UNITS, input_shape=(RESHAPED, INPUT_DIM)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(OUTPUT_SIZE),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(OUTPUT_SIZE),
        ]
    )
    return rnn_model


def model_compile(model):
    return model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, decay=LEARNING_RATE * 0.1),
        metrics=["accuracy"],
    )
