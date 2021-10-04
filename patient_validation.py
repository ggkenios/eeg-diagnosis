import numpy as np
import pandas as pd
from os import listdir

from common import *

# Load trained model
model = model_compile(model_build())
model.load_weights(f"{PATH}checkpoint.h5")

x = np.load(f"{PATH}x_data.npy")
y = np.load(f"{PATH}y_data.npy")
x = x.reshape((len(y), RESHAPED, INPUT_DIM))

for label in CLASS_LIST:
    for file in listdir(f"{PATH_CLOSED}{label}/"):

        # Read pickle files as Dataframes iteratively
        df = pd.DataFrame(pd.read_pickle(f"{PATH_CLOSED}{label}/{file}"), columns=CHANNELS)
        length = len(df)
        data = []
        if length < CUTS_NUMBER:
            pass
        else:
            # Cut into batches, append into a numpy array of shape (-1, 1000, 19) and count to create labels
            length = len(df)
            for batch in range(int(length/CUTS_NUMBER)):
                cut = df[CUTS_NUMBER*batch: CUTS_NUMBER*(batch+1)]
                data.append(cut.to_numpy())

