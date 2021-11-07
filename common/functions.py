import numpy as np
import tensorflow as tf
from tensorflow.keras import models, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, Activation
import matplotlib.pyplot as plt

from common.constants import UNITS, TIME_POINTS, NUMBER_OF_CHANNELS, OUTPUT_SIZE, LEARNING_RATE, RESHUFFLE, BATCH_SIZE


########################
#  1. Model Functions  #
########################

def model_build():
    """Building an RNN model

    Returns:
        Keras sequential model
    """

    rnn_model = models.Sequential(
        [
            tf.keras.layers.Conv1D(UNITS, 3, input_shape=(TIME_POINTS, NUMBER_OF_CHANNELS)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            LSTM(UNITS),
            BatchNormalization(),
            Dense(16, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Dense(OUTPUT_SIZE, activation='softmax'),
        ]
    )
    return rnn_model


def model_build_2():
    """Building an RNN model

    Returns:
        Keras sequential model
    """

    rnn_model = models.Sequential(
        [
            LSTM(UNITS, input_shape=(TIME_POINTS, NUMBER_OF_CHANNELS)),
            BatchNormalization(),
            Dense(OUTPUT_SIZE, activation='softmax'),
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


########################
#  2. Data processing  #
########################

def train_test_set_split(x_array: np.array, y_array: np.array, dic: dict, *args: int) -> (
        np.array, np.array, np.array, np.array
):
    """Get the train test split by having as input the data and patient IDs to include in the training set.

    Args:
        x_array (numpy.array): EEG data numpy array (3D)
        y_array (numpy.array): Label data numpy array (1D)
        dic (dict): Dictionary with count of 2-second batches per patient
        *args (int): Patient IDs of the patients to include in the test set.

    Returns:
        x_train: Train split of our data.
        x_test: Test split of our data.
        y_train: Train split of label data.
        y_test: Test split of label data.
    """
    # 1-hot-encode labels
    y_array = to_categorical(y_array)

    # First, we create a dictionary
    # Key: The starting data point of a specific patient
    # Value: The final data point of the specific patient
    dic_range = {}
    for patient_id in args:
        summation = 0
        for previous_patient_ids in range(0, patient_id):
            summation += dic[previous_patient_ids]
        dic_range[summation] = summation + dic[patient_id]

    # Then we create the a list with all the indexes for these patients
    index_list = []
    for k, v in dic_range.items():
        index_list = index_list + list(range(k, v))

    # Finally create a train-test data split.
    x_test = x_array[index_list]
    y_test = y_array[index_list]
    x_train = np.delete(x_array, index_list, axis=0)
    y_train = np.delete(y_array, index_list, axis=0)

    return x_train, x_test, y_train, y_test


def tensor_preparation(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> (
        object, object
):
    """Transforms split data to tensorflow datasets, shuffling them and cut into batches.

    Args:
        x_train: features train data
        x_test: features test data
        y_train: label train data
        y_test: label test data

    Returns:
        Two tensorflow datasets. The train and validation datasets.
    """
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(buffer_size=y_train.shape[0]+y_test.shape[0], seed=1, reshuffle_each_iteration=RESHUFFLE)\
        .batch(BATCH_SIZE)
    validation = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train, validation


################
#  3. Results  #
################

def plot_curves(history: tf.keras.callbacks.History, metrics: list) -> None:
    """Plot metrics

    Args:
        history: history of model.fit function.
        metrics: list of metrics to be plotted.
    """

    n_rows = 1
    n_cols = 2
    fig = plt.figure(figsize=(10, 5))

    for idx, key in enumerate(metrics):
        fig.add_subplot(n_rows, n_cols, idx + 1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


def majority_vote(all_predictions: list, dic: dict, k: int, i: int, y: np.ndarray, z: np.ndarray) -> None:
    """Creates a majority vote prediction.

    Args:
        all_predictions: List. Eg: [0, 5, 3] out of the 8 2-second data for a specific patient 0 were Healthy,
                         5 MCI, and 3 AD.
        dic: A dictionary with 9 key value pairs. We add +1 on the corresponding value, based on
             what the Label was (0, 1 or 2) and what the prediction was (0, 1 or 2).
        k: An iterator to keep track of Patient's ID for each 2-second window.
        i: An iterator to keep track of the number of 2-second window.
        y: Numpy array of all labels for each 2-second window.
        z: Numpy array of all patient's IDs for each 2-second window.
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
