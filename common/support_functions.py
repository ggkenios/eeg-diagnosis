import tensorflow as tf
from tensorflow.keras import layers, models, losses

from common.constants import UNITS, TIME_POINTS, NUMBER_OF_CHANNELS, OUTPUT_SIZE, LEARNING_RATE


def model_build():
    """Creating an RNN model"""

    # Model
    rnn_model = models.Sequential(
        [
            layers.LSTM(UNITS, input_shape=(TIME_POINTS, NUMBER_OF_CHANNELS)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(OUTPUT_SIZE, activation='softmax'),
        ]
    )
    return rnn_model


def model_compile(model):
    return model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, decay=LEARNING_RATE * 0.1),
        metrics=["accuracy"],
    )
