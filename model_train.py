import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from common import *


# Get data
x = np.load(f"{PATH}x_data.npy")
y = np.load(f"{PATH}y_data.npy")

# Train-Test Split
x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    stratify=y,
    test_size=TEST_SIZE,
    random_state=1,
)

# To tensorflow tensors
x_train = tf.convert_to_tensor(x_train)
x_val = tf.convert_to_tensor(x_val)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)

# Build and compile the model
model = model_build()
model_compile(model)

# Start training
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=checkpoint_acc,
)
