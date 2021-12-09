import numpy as np
from tensorflow.keras.utils import to_categorical

from common import (
    PATH,
    EPOCHS,
    BATCH_SIZE,
    tensor_preparation,
    model_build,
    model_compile,
    lr_reducer,
    checkpoints,
    plot_curves
)


# Get data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")

# Train-Test Split
FOLDS = 5

for fold in range(FOLDS):
    val_index = []
    for j in range(int(x.shape[0]/FOLDS)):
        val_index = val_index + [5*j+ fold]

    x_train = x
    x_test = x[val_index]
    x_train = np.delete(x_train, val_index, axis=0)

    y_train = y
    y_test = y[val_index]
    y_train = np.delete(y_train, val_index, axis=0)

    # 1-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

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
        callbacks=[checkpoints("s", fold), lr_reducer],
    )

    # Accuracy - loss plots
    plot_curves(history, ['accuracy', 'loss'])
