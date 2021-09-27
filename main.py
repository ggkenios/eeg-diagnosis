##################
#  1. Libraries  #

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from extra.constants import PATH


#############
#  2. Code  #

x = np.load(f"{PATH}x_data.npy")
y = np.load(f"{PATH}y_data.npy")

x_train, x_val, y_train, y_val = train_test_split(
    x,
    y,
    stratify=y,
    test_size=0.2,
    random_state=1)

model = keras.Sequential()
model.add(layers.Embedding(input_dim=3800, output_dim=256))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(3))

model.summary()