import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

from common.constants import BATCH_SIZE, RESHUFFLE


def get_patient_indexes(z_array: np.array, *args: int) -> list:
    """It gets the indexes for the required patients, by ID

    Args:
        z_array: Patient ID data numpy array (1D)
        *args  : Patient ID integers of the patients to get the indexes for.

    Returns:
        A list with all the indexes for the given patients
    """

    # Get the number of segments for each patient
    unique, counts = np.unique(z_array, return_counts=True)
    patient_counts = dict(zip(unique, counts))

    # First, we create a dictionary
    # Key: The starting data point of a specific patient
    # Value: The final data point of the specific patient
    dic_range = {}
    for patient_id in args:
        summation = 0
        for previous_patient_ids in range(0, patient_id):
            summation += patient_counts[previous_patient_ids]
        dic_range[summation] = summation + patient_counts[patient_id]

    # Then we create the a list with all the indexes for these patients
    index_list = []
    for k, v in dic_range.items():
        index_list = index_list + list(range(k, v))

    return index_list


def train_test_patient_split(x_array: np.array, y_array: np.array, z_array: np.array, *args: int) -> (
        np.array, np.array, np.array, np.array
):
    """Get the train test split by having as input the data and patient IDs to include in the training set.

    Args:
        x_array: EEG data numpy array (3D)
        y_array: Label data numpy array (1D)
        z_array: Patient ID data numpy array (1D)
        *args  : Patient IDs of the patients to include in the test set.

    Returns:
        x_train: Train split of our data.
        x_test : Test split of our data.
        y_train: Train split of label data.
        y_test : Test split of label data.
    """

    # 1-hot-encode labels
    y_array = to_categorical(y_array)

    # Get patient indexes to exclude
    index_list = get_patient_indexes(z_array, *args)

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
        x_test : features test data
        y_train: label train data
        y_test : label test data

    Returns:
        Two tensorflow datasets. The train and validation datasets.
    """

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=y_train.shape[0] + y_test.shape[0], seed=1, reshuffle_each_iteration=RESHUFFLE)
        .batch(BATCH_SIZE)
    )
    validation = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train, validation
