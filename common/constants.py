from os import path, makedirs
import tensorflow as tf

# Model
MODEL = "conv_lstm"  # In ["lstm", "conv_blstm", "conv_lstm"]

# Model hyper-parameters
LEARNING_RATE = 0.001
EPOCHS_SEGMENT = 40
EPOCHS_PATIENT = 15
BATCH_SIZE = 32

UNITS = 512
RESHUFFLE = True

# Cut data into segments
TIME_POINTS = 1000   # 500 equal to 1 second

# Train-Test Ratio
VALIDATION_SIZE = 0.2

# Paths
PATH = "/Users/georgiosgkenios/Movies/thesis"
PATH_DATA = f"{PATH}/data"                # Here are the raw data
PATH_REPORTS = f"{PATH}/reports"          # Here the reports will be stored
PATH_CHECKPOINTS = f"{PATH}/checkpoints"  # Here the checkpoints will be stored

# Classes
CLASS_H = "Healthy"  # Class 0
CLASS_MCI = "MCI"    # Class 1
CLASS_AD = "AD"      # Class 2
CLASS_LIST = [CLASS_H, CLASS_MCI, CLASS_AD]
CLASS_NUMBER = len(CLASS_LIST)

# Reducer
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1**(1/2),
    mode='min',
    patience=3,
    min_lr=0.5e-6,
    verbose=1
)

# Other
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
NUMBER_OF_CHANNELS = len(CHANNELS)
SHAPE = (TIME_POINTS, NUMBER_OF_CHANNELS)

# Create directories if they do not exist
PATH_LIST = [PATH, PATH_DATA, PATH_REPORTS, PATH_CHECKPOINTS]
for directory in PATH_LIST:
    if not path.exists(directory):
        makedirs(directory)


