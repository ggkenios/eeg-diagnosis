##################
#  1. Libraries  #
import pandas as pd
from os import listdir, mkdir

from extra.constants import PATH_CLOSED, PATH_CUT, CLASS_H, CLASS_MCI, CLASS_AD, CHANNELS, BATCH_NUMBER


#############
#  2. Code  #

for label in [CLASS_AD, CLASS_H, CLASS_MCI]:
    for file in listdir(f"{PATH_CLOSED}{label}/"):

        # Read pickle files as Dataframes iteratively
        df = pd.DataFrame(pd.read_pickle(f"{PATH_CLOSED}{label}/{file}"), columns=CHANNELS)
        length = len(df)

        if length < BATCH_NUMBER:
            pass
        else:
            # Create new directories for the batches
            try:
                mkdir(f"{PATH_CUT}{label}/{file.split('.')[0]}")
            except FileExistsError:
                print(f"Directory already exists: {PATH_CUT}{label}/{file.split('.')[0]}")

            # Cut into batches and write in CSV format
            length = len(df)
            for batch in range(int(length/1000)):
                exec(f"df_{batch} = df[BATCH_NUMBER*batch: BATCH_NUMBER*(batch+1)]")
                exec(f"df_{batch}.to_csv(f'{PATH_CUT}{label}/{file.split('.')[0]}/{batch}_{file.split('.')[0]}.csv', index=False)")
