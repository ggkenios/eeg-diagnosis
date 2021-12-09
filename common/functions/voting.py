import numpy as np
import pandas as pd

from common.constants import CLASS_NUMBER, PATH_REPORTS, SHAPE
from common.functions.plots import confusion_matrix_dict, plot_confusion_matrix
from common.functions.data_processing import get_patient_indexes


def voting(model, x_array: np.array, y_array: np.array, z_array: np.array, *args) -> None:
    """Calculates hard and soft voting for each patient and plots 2 confusion matrices.

    Args:
        model  : Compiled tensorflow model
        x_array: EEG data numpy array (3D)
        y_array: Label data numpy array (1D)
        z_array: Patient ID data numpy array (1D)
        *args  : Integers that denote patient ID.
    """
    # Create a report
    col = ["patient", "label", "soft_label", "hard_label"]
    for i in range(CLASS_NUMBER):
        col.append(f"soft_array_{i}")
        col.append(f"hard_array_{i}")
    df = pd.DataFrame(columns=col)

    # Create empty dictionaries to plot the 2 confusion matrices
    soft_confusion = confusion_matrix_dict(CLASS_NUMBER)
    hard_confusion = confusion_matrix_dict(CLASS_NUMBER)

    for arg in args:
        # Get the indexes
        indexes = get_patient_indexes(z_array, arg)

        # Array of zeros to fill later with softmax
        soft_predict = np.zeros((len(indexes), CLASS_NUMBER))
        # Array to count prediction label for each segment
        hard_predict = CLASS_NUMBER * [0]

        c = 0
        for index in indexes:
            softmax = model.predict(x_array[index].reshape((1,) + SHAPE))
            soft_predict[c] = softmax
            predict_label = np.argmax(softmax)
            hard_predict[predict_label] += 1
            c += 1

        # Calculate mean of softmax for the current patient
        soft_mean = np.mean(soft_predict, axis=0)

        # Soft and hard prediction labels
        soft_prediction = np.argmax(soft_mean)
        hard_prediction = np.argmax(hard_predict)
        true_label = y_array[indexes[0]]

        # Add 1 to the confusion matrix based on the predictions
        soft_confusion[f"t_{true_label}_{soft_prediction}"] += 1
        hard_confusion[f"t_{true_label}_{hard_prediction}"] += 1

        print(
            f"Label            :  {true_label}   {'||'}   Patient:  {arg} / {len(set(z_array))}",
            f"Soft Voting Label:  {soft_prediction}   {'||'}   Soft Voting Array:  {soft_mean}",
            f"Hard Voting Label:  {hard_prediction}   {'||'}   Hard Voting Array:  {hard_predict}",
            f"------------",
            sep="\n",
        )

        # Append the data to the report dataframe
        dictionary = {
            "patient": arg,
            "label": true_label,
            "soft_label": soft_prediction,
            "hard_label": hard_prediction,
        }
        for i in range(CLASS_NUMBER):
            dictionary[f"soft_array_{i}"] = soft_mean[i]
            dictionary[f"hard_array_{i}"] = hard_predict[i]
        df = df.append(dictionary, ignore_index=True)

    # Export the report
    if len(args) < 10:
        track = args
    else:
        track = "all"

    df.to_csv(f"{PATH_REPORTS}/voting_{track}.csv", index=False)

    # Plot the 2 confusion matrices
    plot_confusion_matrix(soft_confusion, "Soft Voting Confusion Matrix")
    plot_confusion_matrix(hard_confusion, "Hard Voting Confusion Matrix")
