import numpy as np
import pandas as pd
from os import listdir

from common import PATH, PATH_CLOSED, CLASS_LIST, CHANNELS, TIME_POINTS


# Track X: Data
data = []
# Track Y: Labels
counts = [0, 0, 0]
# Track Z: Patients
c = 0
patients = []

for label in CLASS_LIST:
    for file in listdir(f"{PATH_CLOSED}{label}/"):

        # Read pickle files as Dataframes iteratively
        df = pd.DataFrame(pd.read_pickle(f"{PATH_CLOSED}{label}/{file}"), columns=CHANNELS)
        length = len(df)

        if length >= TIME_POINTS:
            # Cut into batches, append into a numpy array of shape (-1, TIME_POINTS, 19) and count to create labels
            for batch in range(int(length / TIME_POINTS)):
                cut = df[TIME_POINTS * batch: TIME_POINTS * (batch + 1)]
                data.append(cut.to_numpy())
                counts[CLASS_LIST.index(label)] += 1
                patients.append(c)
            c += 1

# To numpy arrays
x = np.array(data)
y = np.array([0] * counts[0] + [1] * counts[1] + [2] * counts[2])
z = np.array(patients)

# Save data as .npy files
np.save(f"{PATH}x_data.npy", x)
np.save(f"{PATH}y_data.npy", y)
np.save(f"{PATH}z_data.npy", z)
