import numpy as np
import pandas as pd
from os import listdir

from common import *

# Load trained model
model = model_build()
model_compile(model)
model.load_weights(f"{PATH}checkpoint.h5")


for label in CLASS_LIST:
    for file in listdir(f"{PATH_CLOSED}{label}/"):

        # Read pickle files as Dataframes iteratively
        df = pd.DataFrame(pd.read_pickle(f"{PATH_CLOSED}{label}/{file}"), columns=CHANNELS)
        length = len(df)

        data = []
        all_predictions = [0, 0, 0]

        if length < CUTS_NUMBER:
            pass
        else:
            # Cut into batches, append into a numpy array of shape (-1, 1000, 19) and count to create labels
            length = len(df)
            for batch in range(int(length/CUTS_NUMBER)):
                cut = df[CUTS_NUMBER*batch: CUTS_NUMBER*(batch+1)]
                data.append(cut.to_numpy())

            # Predict for every 2-second piece
            for piece in data:
                result = model.predict(piece.reshape(1, RESHAPED, INPUT_DIM))
                predict_label = np.argmax(result, axis=-1)
                predicted = int(str(predict_label)[1])
                all_predictions[predicted] += 1

            # Majority vote for each patient
            prediction = all_predictions.index(max(all_predictions))
            print(prediction)

            # Create the 3x3 confusion matrix
            if label == CLASS_LIST[0]:
                if prediction == 0:
                    dic["t0_p0"] += 1
                elif prediction == 1:
                    dic["t0_p1"] += 1
                elif prediction == 2:
                    dic["t0_p2"] += 1
            if label == CLASS_LIST[1]:
                if prediction == 0:
                    dic["t1_p0"] += 1
                elif prediction == 1:
                    dic["t1_p1"] += 1
                elif prediction == 2:
                    dic["t1_p2"] += 1
            if label == CLASS_LIST[2]:
                if prediction == 0:
                    dic["t2_p0"] += 1
                elif prediction == 1:
                    dic["t2_p1"] += 1
                elif prediction == 2:
                    dic["t2_p2"] += 1

print(dic)
