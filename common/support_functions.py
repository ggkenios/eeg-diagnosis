import tensorflow as tf
from tensorflow.keras import layers, models, losses

from common.constants import UNITS, TIME_POINTS, NUMBER_OF_CHANNELS, OUTPUT_SIZE, LEARNING_RATE


def model_build():
    """Building an RNN model"""

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
    """Compile the model

    Arg:
        model (class): Tensorflow model
    """

    return model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, decay=LEARNING_RATE * 0.1),
        metrics=["accuracy"],
    )


def majority_vote(all_predictions: list, dic: dict):
    """Creates a majority vote prediction. Not easy to understand

    Some extra info needed are:
    i: Iterator. In range from (0, 4502), with 4502 being the number of 2-second data cuts.
    y[i]: Is the label for the i-th data cut.

    Args:
        all_predictions (list): [0, 5, 3] e.g. Which means, out of the 8 2-second data for a specific patient
                                0 were classified as Healthy, 5 as MCI, and 3 as AD

        dic (dict): A dictionary with 9 key value pairs. We add +1 on the corresponding value, based on
                    what the Label was (0, 1 or 2) and what the prediction was (0, 1 or 2).
    """

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
