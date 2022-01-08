import numpy as np

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
)


# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

patient_out = [2, 10, 15, 20, 25, 30, 37, 42, 51]

# Split dataset by patient (2.2 function)
(x_train,
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
    epochs=EPOCHS,
    callbacks=[checkpoints("p"), lr_reducer],
)

# Create confusion matrix
model.load_weights(f"{PATH_CHECKPOINTS}/p_-1.h5")
voting(model, x, y, z, *patient_out)
