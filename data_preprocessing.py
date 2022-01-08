import mne
import numpy as np
import pandas as pd
from os import listdir

from common import PATH, PATH_DATA, CLASS_LIST, CHANNELS, TIME_POINTS, CLASS_NUMBER, PATH_REPORTS

# Track X: Data
data = []
# Track Y: Labels
counts = CLASS_NUMBER * [0]
# Track Z: Patients
c = 0
patients = []

# Report of the dataset
col = ["Patient", "Start", "End", "Class"]
df = pd.DataFrame(columns=col)

for label in CLASS_LIST:
    for patient in listdir(f"{PATH_DATA}/{label}"):
        file_path = f"{PATH_DATA}/{label}/{patient}"

        # Read the edf and get the annotations
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.pick_channels(CHANNELS)
        # raw.notch_filter(freqs=50)
        raw.filter(l_freq=0.1, h_freq=42)

        # ica = mne.preprocessing.ICA(n_components=19, random_state=0)
        # ica.fit(raw)
        # ica.apply(raw)
        anno = mne.events_from_annotations(raw)

        # Here we want to find starting and ending point for closed eyes.
        if patient[:2] == "FA":
            for key in anno[1].keys():
                if "a1+a2 on" in key.lower():
                    for i in range(anno[0].shape[0]):
                        if anno[0][i][2] == anno[1][key]:
                            start = anno[0][i][0]
                            end = start + 150000
                            break

        elif patient[:2] == "00":
            for key in anno[1].keys():
                if "eyes closed" in key.lower():
                    for i in range(anno[0].shape[0]):
                        if anno[0][i][2] == anno[1][key]:
                            start = anno[0][i][0]
                            end = start + 150000
                            break

        array = np.array(raw.to_data_frame().iloc[:, 1:])[start: end]
        inverse_array = np.transpose(array)

        length = array.shape[0]

        '''
        outliers = {}

        for channel in range(NUMBER_OF_CHANNELS):
            deviation = np.std(inverse_array[channel])
            mean = np.mean(inverse_array[channel])
            outliers[channel] = {"mean": mean, "std": deviation}
        '''

        # Cut into batches, append into a numpy array of shape (-1, TIME_POINTS, CHANNELS) and count to create labels
        for batch in range(int(length / TIME_POINTS)):
            break_boolean = False
            cut = array[TIME_POINTS * batch: TIME_POINTS * (batch + 1)]
            inverse = np.transpose(cut)
            '''
            for channel in range(NUMBER_OF_CHANNELS):
                for value in inverse[channel]:
                    if value >= outliers[channel]["mean"] + 4*outliers[channel]["std"] or value <= outliers[channel]["mean"] - 4*outliers[channel]["std"]:
                        break_boolean = True
            if break_boolean == True:
              continue
            '''
            for number, channel in enumerate(inverse):
                inverse[number] = np.fft.fft(channel)
            cut = np.transpose(inverse)
            data.append(cut)
            counts[CLASS_LIST.index(label)] += 1
            patients.append(c)
        c += 1

        dictionary = {
            "Patient": patient.replace(".edf", ""),
            "Start": start / 500,
            "End": end / 500,
            "Class": label,
        }
        df = df.append(dictionary, ignore_index=True)

x = np.array(data)
y = np.array([0] * counts[0] + [1] * counts[1] + [2] * counts[2])
z = np.array(patients)

# Save data as .npy files
np.save(f"{PATH}/x_data.npy", x)
np.save(f"{PATH}/y_data.npy", y)
np.save(f"{PATH}/z_data.npy", z)

df.to_csv(f"{PATH_REPORTS}/dataset.csv", index=False)
