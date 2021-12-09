import numpy as np

from common import PATH_CHECKPOINTS, PATH, model_build, model_compile, voting

# Read the data
x = np.load(f"{PATH}/x_data.npy")
y = np.load(f"{PATH}/y_data.npy")
z = np.load(f"{PATH}/z_data.npy")

# Model
model = model_build()
model_compile(model)
model.load_weights(f"{PATH_CHECKPOINTS}/p_-1.h5")

FOLDS = 6

for fold in range(FOLDS):
    model.load_weights(f"{PATH_CHECKPOINTS}/p_{fold}.h5")

    patient_out = [j for j in range(len(np.unique(z))) if (j + fold) % FOLDS == 0]

    voting(model, x, y, z, *patient_out)
