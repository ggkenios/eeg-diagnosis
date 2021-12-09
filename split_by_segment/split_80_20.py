import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from common import (
    PATH,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SIZE,
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
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    stratify=y,
    test_size=VALIDATION_SIZE,
    random_state=1,
    )

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
    callbacks=[checkpoints("s"), lr_reducer],
)

# Accuracy - loss plots
plot_curves(history, ['accuracy', 'loss'])
