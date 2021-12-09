import tensorflow as tf
from tensorflow.keras import models, losses
from tensorflow.keras.regularizers import L1
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Conv1D,
    Dropout,
    Activation,
    BatchNormalization,
)

from common.constants import UNITS, SHAPE, CLASS_NUMBER, LEARNING_RATE, PATH_CHECKPOINTS


def model_build():
    """Building a model

    Returns:
        Keras sequential model
    """

    rnn_model = models.Sequential(
        [
            Conv1D(UNITS, 3, input_shape=SHAPE),
            BatchNormalization(),
            Activation('relu'),
            LSTM(UNITS),
            BatchNormalization(),
            Dense(16, kernel_regularizer=L1(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Dense(CLASS_NUMBER, activation='softmax'),
        ]
    )
    return rnn_model


def model_build_2():
    """Building a model

    Returns:
        Keras sequential model
    """

    rnn_model = models.Sequential(
        [
            LSTM(UNITS, input_shape=SHAPE),
            BatchNormalization(),
            Dense(16),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Dense(CLASS_NUMBER, activation='softmax'),
        ]
    )
    return rnn_model


def model_compile(model) -> None:
    """Compile the model

    Args:
        model: Tensorflow keras sequential model
    """

    return model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, decay=LEARNING_RATE * 0.1),
        metrics=["accuracy"],
    )


def checkpoints(split_type: str, fold: int = -1) -> tf.keras.callbacks.ModelCheckpoint:
    """Creates a checkpoints storing best performing val accuracy.

    The checkpoint file has a name: "{split_type}_{fold}.h5".

    Args:
        split_type: A string to denote whether split is by Segment (s) or Patient (p).
        fold      : The number of the fold during cross validation. Default is -1
                    when there is no cross validation.

    Returns:
        A tf.keras.callbacks.ModelCheckpoint class that stores the best val accuracy.
    """

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{PATH_CHECKPOINTS}/{split_type}_{fold}.h5",
        monitor='val_accuracy',
        save_best_only=True
        )

    return checkpoint
