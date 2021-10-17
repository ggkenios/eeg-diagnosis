import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

    # Build and compile the model
    model = model_build()
    model_compile(model)

    # Start training
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_acc, lr_reducer],
    )

    # Accuracy plot
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='best')
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()
