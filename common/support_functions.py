import numpy
import tensorflow as tf
from tensorflow.keras import layers, models, losses

from common.constants import UNITS, TIME_POINTS, NUMBER_OF_CHANNELS, OUTPUT_SIZE, LEARNING_RATE


def model_build():
    """Building an RNN model

    Returns:
        class: Keras sequential model

    """

    # Model
    rnn_model = models.Sequential(
        [
            layers.Bidirectional(layers.LSTM(UNITS, return_sequences=True), input_shape=(TIME_POINTS, NUMBER_OF_CHANNELS)),
            layers.Bidirectional(layers.LSTM(int(UNITS/2))),
            layers.BatchNormalization(),
            layers.Dense(OUTPUT_SIZE, activation='softmax'),
        ]
    )
    return rnn_model


def model_compile(model):
    """Compile the model

    Arg:
        model (class): Tensorflow keras model

    Returns:
        class: Compiled model

    """

    return model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, decay=LEARNING_RATE * 0.1),
        metrics=["accuracy"],
    )


def majority_vote(all_predictions: list, dic: dict, k: int, i: int, y: numpy.ndarray, z: numpy.ndarray):
    """Creates a majority vote prediction. Not easy to understand

    Some extra info needed are:
    i: Iterator. In range from (0, 4502), with 4502 being the number of 2-second data cuts.
    y[i]: Is the label for the i-th data cut.

    Args:
        all_predictions (list): [0, 5, 3] e.g. out of the 8 2-second data for a specific patient 0 were Healthy,
                                5 MCI, and 3 AD
        dic (dict): A dictionary with 9 key value pairs. We add +1 on the corresponding value, based on
                    what the Label was (0, 1 or 2) and what the prediction was (0, 1 or 2).
        k (int): An iterator to keep track of Patient's ID for each 2-second window
        i (int): An iterator to keep track of the number of 2-second window
        y (numpy.ndarray): Numpy array of all labels for each 2-second window
        z (numpy.ndarray): Numpy array of all patient's IDs for each 2-second window
    """

    prediction = all_predictions.index(max(all_predictions))
    print("Prediction: ", prediction, "  ||   Patient: ", k + 1, "/", len(set(z)))

    # Add +1 on dictionary values to form the 3x3 confusion matrix
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
