import numpy as np

from common import (
    PATH,
    BATCH_SIZE,
    EPOCHS,
    checkpoint_acc,
    lr_reducer,
    model_build_2,
    model_compile,
    train_test_set_split,
    plot_curves,
    tensor_preparation,
)


# Read the data
x = np.load(f"{PATH}x_data.npy")
y = np.load(f"{PATH}y_data.npy")
z = np.load(f"{PATH}z_data.npy")

unique, counts = np.unique(z, return_counts=True)
patient_counts = dict(zip(unique, counts))

# Run the function from 1.3 Support Functions to split dataset
(x_train,
 x_test,
 y_train,
 y_test,
 ) = train_test_set_split(x, y, patient_counts, 8, 12, 15, 30, 40, 42, 44, 45, 47, 48, 49)

# To tensorflow dataset
train, validation = tensor_preparation(x_train, x_test, y_train, y_test)

# Build and compile the model
model = model_build_2()
model_compile(model)

# Start training
history = model.fit(
    train,
    validation_data=validation,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_acc, lr_reducer],
)

# Accuracy - loss plots
plot_curves(history, ['accuracy', 'loss'])
