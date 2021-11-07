import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from common import *


# Load trained model
model = model_build()
model_compile(model)

cp_path = f"{PATH}/Checkpoints/Patient split/checkpoint.h5"
model.load_weights(cp_path)

# Load data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

# Variables
all_predictions = [0, 0, 0]
i = 0  # Track iteration
k = 0  # Track patient

# Iterate through data, and make a majority vote prediction for each patient
while i < len(y):
    if z[i] == k:
        result = model.predict(x[i].reshape(1, TIME_POINTS, NUMBER_OF_CHANNELS))
        predict_label = np.argmax(result, axis=-1)
        predicted = int(str(predict_label)[1])
        all_predictions[predicted] += 1

        i += 1

    else:
        majority_vote(all_predictions, dic, k, i, y, z)
        all_predictions = [0, 0, 0]
        k += 1
# After the final 2-second batch is predicted for the last patient, we take majority vote, as else won't be reached
majority_vote(all_predictions, dic, k, i, y, z)

# Results
print(dic)

array = [
    [dic["t0_p0"], dic["t0_p1"], dic["t0_p2"]],
    [dic["t1_p0"], dic["t1_p1"], dic["t1_p2"]],
    [dic["t2_p0"], dic["t2_p1"], dic["t2_p2"]],
]

df_cm = pd.DataFrame(
    array,
    index=CLASS_LIST,
    columns=CLASS_LIST,
)

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True)
plt.show()
