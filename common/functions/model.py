import tensorflow as tf
from tensorflow.keras import models, losses
from tensorflow.keras.regularizers import L1
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Conv1D,
    Dropout,
    Activation,
    Bidirectional,
    BatchNormalization,
)

from common.constants import UNITS, SHAPE, CLASS_NUMBER, LEARNING_RATE, PATH_CHECKPOINTS


def lstm():
    """Building an RNN model with an LSTM layer.

    Returns:
        Keras sequential model.
    """

    rnn_model = models.Sequential(
        [
            LSTM(units=512, stateful=False, input_shape=SHAPE, name="LSTM_512"),
            BatchNormalization(name="Batch_Normalization_LSTM"),
            Dense(units=16, kernel_regularizer=L1(0.001), name="Dense_16"),
            BatchNormalization(name="Batch_Normalization_Dense"),
            Activation('relu', name="ReLU_Dense"),
            Dropout(0.2, name="Dropout_0.2"),
            Dense(CLASS_NUMBER, activation='softmax', name="Dense_3"),
        ]
    )
    return rnn_model


def conv_blstm():
    """Building an RNN model, with convolutional and bidrectional LSTM layers.

    Returns:
        Keras sequential model.
    """

    rnn_model = models.Sequential(
        [
            Conv1D(filters=512, kernel_size=3, strides=1, input_shape=SHAPE, name="Conv1D_512"),
            BatchNormalization(name="Batch_Normalization_Conv1D"),
            Activation('relu', name="ReLU_Conv1D"),
            Bidirectional(LSTM(units=UNITS, stateful=False, name="LSTM_512"), name="Bidirectional"),
            BatchNormalization(name="Batch_Normalization_BLSTM"),
            Dense(units=16, kernel_regularizer=L1(0.001), name="Dense_16"),
            BatchNormalization(name="Batch_Normalization_Dense"),
            Activation('relu', name="ReLU_Dense"),
            Dropout(0.2, name="Dropout_0.2"),
            Dense(CLASS_NUMBER, activation='softmax', name="Dense_3"),
        ]
    )
    return rnn_model


def conv_lstm():
    """Building an RNN model, with conv1d and LSTM layers.

    Returns:
        Keras sequential model.
    """

    rnn_model = models.Sequential(
        [
            Conv1D(filters=512, kernel_size=3, strides=1, input_shape=SHAPE, name="Conv1D_512"),
            BatchNormalization(name="Batch_Normalization_Conv1D"),
            Activation('relu', name="ReLU_Conv1D"),
            LSTM(units=UNITS, stateful=False, name="LSTM_512"),
            BatchNormalization(name="Batch_Normalization_LSTM"),
            Dense(units=16, kernel_regularizer=L1(0.001), name="Dense_16"),
            BatchNormalization(name="Batch_Normalization_Dense"),
            Activation('relu', name="ReLU_Dense"),
            Dropout(0.2, name="Dropout_0.2"),
            Dense(CLASS_NUMBER, activation='softmax', name="Dense_3"),
        ]
    )
    return rnn_model


def model_compile(model) -> None:
    """Compile a keras model.

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
