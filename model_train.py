import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from common import *


if __name__ == "__main__":

    # Get data
    x = np.load(f"{PATH}x_data.npy")
    y = np.load(f"{PATH}y_data.npy")

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        stratify=y,
        test_size=VALIDATION_SIZE,
        random_state=1,
        )

    # To tensorflow tensors
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(to_categorical(y_train, OUTPUT_SIZE))
    y_test = tf.convert_to_tensor(to_categorical(y_test, OUTPUT_SIZE))

    # To tensorflow dataset
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(buffer_size=x.shape[0], seed=1, reshuffle_each_iteration=RESHUFFLE)\
        .batch(BATCH_SIZE)
    validation = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    # Build and compile the model
    model = model_build()
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

