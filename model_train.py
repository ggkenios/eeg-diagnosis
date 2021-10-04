import numpy as np
from sklearn.model_selection import train_test_split

from common import *


# Get data
x = np.load(f"{PATH}x_data.npy")
y = np.load(f"{PATH}y_data.npy")
x = x.reshape((len(y), RESHAPED, INPUT_DIM))

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    stratify=y,
    test_size=TEST_SIZE,
    random_state=1)

# Build and compile the model
model = model_build()
model_compile(model)

# Start training
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=checkpoint_acc,
)
