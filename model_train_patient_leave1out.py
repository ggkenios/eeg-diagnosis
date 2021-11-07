import numpy as np
import pandas as pd

from common import (
    PATH,
    BATCH_SIZE,
    EPOCHS,
    lr_reducer,
    model_build,
    model_compile,
    train_test_set_split,
    tensor_preparation,
)

col = ["Patient_out", "Val_accuracy", "Val_loss"]
df = pd.DataFrame(columns=col)

# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

unique, counts = np.unique(z, return_counts=True)
patient_counts = dict(zip(unique, counts))

for patient_out in range(54):
    # Run the function from 1.3 Support Functions to split dataset
    (x_train,
     x_test,
     y_train,
     y_test,
     ) = train_test_set_split(x, y, patient_counts, patient_out)

    # To tensorflow dataset
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

    # Accuracy - loss plots
    acc = history.history['val_accuracy'][-1]
    loss = history.history['val_loss'][-1]

    dictionary = {
        "Patient_out": patient_out,
        "Val_accuracy": acc,
        "Val_loss": loss,
    }

    df = df.append(dictionary, ignore_index=True)

    # Save the file on each iteration, so we can pick up from where we left, if we stop the code.
    df.to_csv(PATH + "/report_leave1out.csv", index=False)

df.to_csv(PATH + "/report_leave1out_f.csv", index=False)

