import numpy as np
import pandas as pd

from common import (
    voting,
    lr_reducer,
    lstm,
    conv_lstm,
    conv_blstm,
    model_compile,
    train_test_patient_split,
    tensor_preparation,
    checkpoints,
    PATH,
    MODEL,
    EPOCHS_PATIENT,
    BATCH_SIZE,
    PATH_REPORTS,
    PATH_CHECKPOINTS,
)


# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

FOLDS = 6

df = pd.DataFrame()

for fold in range(FOLDS):
    print("------", f"FOLD: {fold}", sep="\n")
    patient_out = [j for j in range(len(np.unique(z))) if (j+fold) % FOLDS == 0]

    # Split dataset by patient (2.2 function)
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_patient_split(x, y, z, *patient_out)

    # Create a shuffled tensorflow dataset (2.2 function)
    train, validation = tensor_preparation(x_train, x_test, y_train, y_test)

    # Build and compile the model
    model = locals()[MODEL]()
    model_compile(model)

    # Start training
    history = model.fit(
        train,
        validation_data=validation,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_PATIENT,
        callbacks=[checkpoints("p", fold), lr_reducer],
    )

    # Plot confusion Matrix for validation samples
    model.load_weights(f"{PATH_CHECKPOINTS}/p_{fold}.h5")
    voting(model, x, y, z, *patient_out)

    # Export reporting
    dictionary = {
        "Patient_out": patient_out,
        "Val_accuracy": history.history['val_accuracy'][-1],
        "Max_val_accuracy": max(history.history['val_accuracy']),
        "Val_loss": history.history['val_loss'][-1],
        "Min_val_loss": min(history.history['val_accuracy'])
    }
    df = df.append(dictionary, ignore_index=True)
    df.to_csv(f"{PATH_REPORTS}/p_fold_{fold}.csv", index=False)

df.to_csv(f"{PATH_REPORTS}/p_fold_finished.csv", index=False)

