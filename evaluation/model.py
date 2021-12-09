import numpy as np

from common import PATH_CHECKPOINTS, PATH, model_build, model_compile, voting

# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

model = model_build()
model_compile(model)
model.load_weights(f"{PATH_CHECKPOINTS}/p_-1.h5")

#patient_out = np.unique(z)
patient_out = [2, 10, 15, 20, 25, 30, 37, 42, 51]

voting(model, x, y, z, *patient_out)
