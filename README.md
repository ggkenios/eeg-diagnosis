# Thesis

### Files
* <a href="https://github.com/Ggkenios/thesis/blob/main/extra/constants.py">Constants</a> <br>
   - Set paths, model, hyperparameters, and other needed constants.
<br>

* <a href="https://github.com/Ggkenios/thesis/blob/main/extra/cut_batches.py">Data Preperation</a> <br>
   - Reading data ilteratively and stores them in 2 numpy files:
      x: Array of dimension (-1, 1000, 19). So, basically a list of 2d arrays: 1000 datapoints (2 seconds) for 19 channels.
      y: Array of labels (-1) for x. The number of cuts for all patients.
<br>
      
* <a href="https://github.com/Ggkenios/thesis/blob/main/model_train.py">Data Preperation</a> <br>
   - Reads the numpy files created from cut_batches.py
   - Splits them into train and test data, 75-25, in a balanced way, in terms of labels in train and test sets.
   - Reshape data to based on timesteps to feed the RNN model.
   - Creates a simple LSTM model with 64 units --> Batch Normalization --> Dense(3)
   - Train it.
