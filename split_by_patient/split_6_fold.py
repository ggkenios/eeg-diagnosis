import numpy as np
import pandas as pd

from common import (
    PATH,
    BATCH_SIZE,
    EPOCHS,
    lr_reducer,
    model_build,
    model_compile,
    train_test_patient_split,
    tensor_preparation,
    checkpoints,
    PATH_REPORTS,
)


# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

FOLDS = 6

col = ["Patient_out", "Val_accuracy", "Max_val_accuracy", "Val_loss", "Min_val_loss"]
df = pd.DataFrame(columns=col)

for fold in range(FOLDS):
    patient_out = [j for j in range(len(np.unique(z))) if (j+fold) % FOLDS == 0]

    # Split dataset by patient (2.2 function)
    (x_train,
     x_test,
     y_train,
     y_test,
     ) = train_test_patient_split(x, y, z, *patient_out)

    # Create a shuffled tensorflow dataset (2.2 function)
    train, validation = tensor_preparation(x_train, x_test, y_train, y_test)

    # Build and compile the model
    model = model_build()
    model_compile(model)

    # Start training
    history = model.fit(
        train,
        validation_data=validation,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoints("p", fold), lr_reducer],
    )

    dictionary = {
        "Patient_out": patient_out,
        "Val_accuracy": history.history['val_accuracy'][-1],
        "Max_val_accuracy": max(history.history['val_accuracy']),
        "Val_loss": history.history['val_loss'][-1],
        "Min_val_loss": min(history.history['val_accuracy'])
    }

    df = df.append(dictionary, ignore_index=True)

    # Save every update, to keep the process from where it stopped
    df.to_csv(f"{PATH_REPORTS}/p_fold_{fold}.csv", index=False)

df.to_csv(f"{PATH_REPORTS}/p_fold_finished.csv", index=False)

