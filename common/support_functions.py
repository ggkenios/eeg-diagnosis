import tensorflow as tf
from tensorflow.keras import layers, models, losses

from common.constants import UNITS, TIME_POINTS, NUMBER_OF_CHANNELS, OUTPUT_SIZE, LEARNING_RATE


def model_build():
    """Creating an RNN model"""

    # Model
    rnn_model = models.Sequential(
        [
            layers.LSTM(int(UNITS*2), input_shape=(TIME_POINTS, NUMBER_OF_CHANNELS)),
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


def majority_vote(all_predictions: list, dic: dict):
    prediction = all_predictions.index(max(all_predictions))
    print("Prediction: ", prediction, "  ||   Patient: ", k + 1, "/", len(set(z)))

    # Create the 3x3 confusion matrix
    if y[i-1] == 0:
        if prediction == 0:
            dic["t0_p0"] += 1
        elif prediction == 1:
            dic["t0_p1"] += 1
        elif prediction == 2:
            dic["t0_p2"] += 1
    elif y[i-1] == 1:
        if prediction == 0:
            dic["t1_p0"] += 1
        elif prediction == 1:
            dic["t1_p1"] += 1
        elif prediction == 2:
            dic["t1_p2"] += 1
    elif y[i-1] == 2:
        if prediction == 0:
            dic["t2_p0"] += 1
        elif prediction == 1:
            dic["t2_p1"] += 1
        elif prediction == 2:
            dic["t2_p2"] += 1
