import tensorflow as tf

# Model hyper-parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 40
UNITS = 512
RESHUFFLE = True

# Cut data into batches
TIME_POINTS = 1000     # 500 equal to 1 second

# Train-Test Ratio
VALIDATION_SIZE = 0.2

# Paths
PATH = "C:/Users/thxsg/Documents/1. Thesis Data"
PATH_DATA = f"{PATH}/my_dataset/"
PATH_CLOSED = f"{PATH}/thesis_closed/"

# Classes
CLASS_H = "Healthy"  # Class 0
CLASS_MCI = "MCI"    # Class 1
CLASS_AD = "AD"      # Class 2
CLASS_LIST = [CLASS_H, CLASS_MCI, CLASS_AD]

# Checkpoints
checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{PATH}/Checkpoints/Normal split/checkpoint.h5",
    monitor='val_accuracy',
    save_best_only=True
)

checkpoint_acc_2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{PATH}/Checkpoints/Patient split/checkpoint.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# Reducer
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1**(1/2),
    mode='min',
    patience=3,
    min_lr=0.5e-6,
    verbose=1
)

# Rest
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
NUMBER_OF_CHANNELS = len(CHANNELS)
OUTPUT_SIZE = len(CLASS_LIST)

dic = {
    "t0_p0": 0,
    "t0_p1": 0,
    "t0_p2": 0,
    "t1_p0": 0,
    "t1_p1": 0,
    "t1_p2": 0,
    "t2_p0": 0,
    "t2_p1": 0,
    "t2_p2": 0,
}
