import numpy as np
import pandas as pd

from common import (
    PATH,
    BATCH_SIZE,
    EPOCHS,
    lr_reducer,
    model_build,
    model_build_2,
    model_compile,
    train_test_set_split,
    tensor_preparation,
)


# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

# Folds for Cross Validation
FOLDS = 6

# Report
col = ["Patient", "Val_accuracy", "Val_loss"]
df = pd.DataFrame(columns=col)

unique, counts = np.unique(z, return_counts=True)
patient_counts = dict(zip(unique, counts))

for i in range(FOLDS):
    patient_out = [j for j in range(len(unique)) if (j+i) % FOLDS == 0]

    # Split dataset by patient (1.4 function)
    (x_train,
     x_test,
     y_train,
     y_test,
     ) = train_test_set_split(x, y, patient_counts, *patient_out)

    # Create a shuffled tensorflow dataset (1.4 function)
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
        callbacks=[lr_reducer],
    )

    dictionary = {
        "Patient_out": patient_out,
        "Val_accuracy": history.history['val_accuracy'][-1],
        "Val_loss": history.history['val_loss'][-1],
    }

    df = df.append(dictionary, ignore_index=True)

    # Save every update, to keep the process from where it stopped
    df.to_csv(f"{PATH}/report_6fold.csv", index=False)

df.to_csv(f"{PATH}/report_6fold_f.csv", index=False)
