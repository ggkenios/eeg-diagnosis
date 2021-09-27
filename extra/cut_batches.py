##################
#  1. Libraries  #

import numpy as np
import pandas as pd
from os import listdir

from extra.constants import PATH, PATH_CLOSED, CLASS_LIST, CHANNELS, BATCH_NUMBER


#############
#  2. Code  #

# Count labels for each class to create label data
counts = [0, 0, 0]
data = []

for label in CLASS_LIST:
    for file in listdir(f"{PATH_CLOSED}{label}/"):

        # Read pickle files as Dataframes iteratively
        df = pd.DataFrame(pd.read_pickle(f"{PATH_CLOSED}{label}/{file}"), columns=CHANNELS)
        length = len(df)

        if length < BATCH_NUMBER:
            pass
        else:
            # Cut into batches, append into a numpy array of shape (-1, 1000, 19) and count to create labels
            length = len(df)
            for batch in range(int(length/1000)):
                cut = df[BATCH_NUMBER*batch: BATCH_NUMBER*(batch+1)]
                data.append(cut.to_numpy())
                counts[CLASS_LIST.index(label)] += 1

x_array = np.array(data)
y_array = np.array([0]*counts[0] + [1]*counts[1] + [2]*counts[2])
np.save(f"{PATH}x_data.npy", x_array)
np.save(f"{PATH}y_data.npy", y_array)
