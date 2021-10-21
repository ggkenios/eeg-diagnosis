import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from common import (
    PATH,
    OUTPUT_SIZE,
    BATCH_SIZE,
    EPOCHS,
    checkpoint_acc,
    lr_reducer,
    model_build,
    model_compile,
    train_test_set,
)

if __name__ == "__main__":

    # Read the data
    x = np.load(f"{PATH}x_data.npy")
    y = np.load(f"{PATH}y_data.npy")
    z = np.load(f"{PATH}z_data.npy")

    unique, counts = np.unique(z, return_counts=True)
    patient_counts = dict(zip(unique, counts))

    # Run the function
    (x_train,
     x_test,
     y_train,
     y_test,
     ) = train_test_set(x, y, patient_counts, 8, 12, 15, 30, 40, 42, 44, 45, 47, 48)

    # To tensorflow tensors
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(to_categorical(y_train, OUTPUT_SIZE))
    y_test = tf.convert_to_tensor(to_categorical(y_test, OUTPUT_SIZE))

    # To tensorflow dataset
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    validation = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)

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
