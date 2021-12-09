import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt

from common.constants import CLASS_NUMBER, CLASS_LIST


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


def confusion_matrix_dict(num_class: int) -> dict:
    """Creates a dictionary with number-of-classes squared keys and values equal to 0.

    Args:
        num_class: An integer equal to the number of classes.

    Returns:
        A dictionary with number-of-classes squared keys and values equal to 0.
        Its keys are like t_{i}_{j} for i the true label and j the predicted one.
    """

    confusion_dict = {}
    for i in range(num_class):
        for j in range(num_class):
            confusion_dict[f"t_{i}_{j}"] = 0

    return confusion_dict


def plot_confusion_matrix(dic: dict, title: str) -> None:
    """Plots a confusion matrix from a dictionary.

    Args:
        dic  : A dictionary that has keys t_{i}_{j} for i true and j prediction
               labels and values the value number of occurrences for each one.
        title: The title of the plot
    """

    array = []
    for i in range(CLASS_NUMBER):
        row = []
        for j in range(CLASS_NUMBER):
            row.append(dic[f"t_{i}_{j}"])
        array.append(row)

    df_cm = pd.DataFrame(
        array,
        index=CLASS_LIST,
        columns=CLASS_LIST,
    )

    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True).set_title(title)
    plt.show()
