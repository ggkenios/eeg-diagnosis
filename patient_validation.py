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
            prediction = all_predictions.index(max(all_predictions))
            print("Prediction: ", prediction, "  ||   Patient: ", k + 1, "/", len(set(z)))

            # Create the 3x3 confusion matrix
            if y[i] == 0:
                if prediction == 0:
                    dic["t0_p0"] += 1
                elif prediction == 1:
                    dic["t0_p1"] += 1
                elif prediction == 2:
                    dic["t0_p2"] += 1
            elif y[i] == 1:
                if prediction == 0:
                    dic["t1_p0"] += 1
                elif prediction == 1:
                    dic["t1_p1"] += 1
                elif prediction == 2:
                    dic["t1_p2"] += 1
            elif y[i] == 2:
                if prediction == 0:
                    dic["t2_p0"] += 1
                elif prediction == 1:
                    dic["t2_p1"] += 1
                elif prediction == 2:
                    dic["t2_p2"] += 1

            all_predictions = [0, 0, 0]
            k += 1

    break

print(dic)

