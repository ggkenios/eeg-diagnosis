import numpy as np

from common import *


# Load trained model
model = model_build()
model_compile(model)
model.load_weights(f"{PATH}checkpoint.h5")

# Load data
x = np.load(f"{PATH}x_data.npy")
y = np.load(f"{PATH}y_data.npy")
z = np.load(f"{PATH}z_data.npy")

# Variables
all_predictions = [0, 0, 0]
i = 0  # Track iteration
k = 0  # Track patient

while True:
    while i < len(y):
        if z[i] == k:
            result = model.predict(x[i].reshape(1, TIME_POINTS, NUMBER_OF_CHANNELS))
            predict_label = np.argmax(result, axis=-1)
            predicted = int(str(predict_label)[1])
            all_predictions[predicted] += 1

            i += 1

        else:
            majority_vote(all_predictions, dic)
            all_predictions = [0, 0, 0]
            k += 1

    majority_vote(all_predictions, dic)
    break

print(dic)
