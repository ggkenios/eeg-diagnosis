# Model hyper-parameters
BATCH_SIZE = 64
EPOCHS = 5
INPUT_DIM = 5
UNITS = 64

# Paths
PATH = "C:/Users/thxsg/Documents/1. Thesis Data/"
PATH_CLOSED = f"{PATH}thesis_closed/"

# Classes
CLASS_H = "Healthy"  # Class 0
CLASS_MCI = "MCI"    # Class 1
CLASS_AD = "AD"      # Class 2
CLASS_LIST = [CLASS_H, CLASS_MCI, CLASS_AD]

# Rest
CUTS_NUMBER = 1000  # 500 rows equal to 1 second

CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

OUTPUT_SIZE = len(CLASS_LIST)
RESHAPED = int(CUTS_NUMBER * len(CHANNELS) / INPUT_DIM)
