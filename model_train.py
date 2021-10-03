import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

from extra.constants import PATH, EPOCHS, BATCH_SIZE, UNITS, RESHAPED, INPUT_DIM, OUTPUT_SIZE, TEST_SIZE


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


def build_model(allow_cudnn_kernel=True):
    """Creating an RNN model"""

    if allow_cudnn_kernel:
        lstm_layer = keras.layers.LSTM(UNITS, input_shape=(RESHAPED, INPUT_DIM))
    else:
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(UNITS), input_shape=(RESHAPED, INPUT_DIM)
        )

    # Model
    rnn_model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(OUTPUT_SIZE),
        ]
    )
    return rnn_model


model = build_model(allow_cudnn_kernel=True)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)


model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)
