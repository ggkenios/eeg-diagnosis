# Thesis

### Common Folder
#### Constants and Functions
* <a href="https://github.com/Ggkenios/thesis/blob/main/common/constants.py">Constants</a> <br>
   - Set paths, model, hyperparameters, and other needed constants.
<br>

* <a href="https://github.com/Ggkenios/thesis/blob/main/common/support_functions.py">Support Functions</a> <br>
   - Functions to: Create model, compile model, calculate majority vote
<br>

#### Majority Vote
* <a href="https://github.com/Ggkenios/thesis/blob/main/common/patient_validation.py">Constants</a> <br>
   - Support code, for <a href="https://github.com/Ggkenios/thesis/blob/main/plots.py">plots.py</a> to calculate majority vote, after the model is trained
<br>
<br>

### <a href="https://github.com/Ggkenios/thesis/blob/main/data_preperation.py">Data Preperation</a> <br>
   * Reading data ilteratively and stores them in 3 numpy files: <br>
     - x: Array of shape (-1, 1000, 19) for patient's data. So, basically a list of 2d arrays: 1000 datapoints (2 seconds-window) for 19 channels. <br>
     - y: Array of shape (-1) that tracks the labels for each 2-second data window. <br>
     - z: Array of shape (-1) that tracks the patient ID for each 2-second data window. <br>

<br>

### <a href="https://github.com/Ggkenios/thesis/blob/main/model_train.py">Model Train</a> <br>
   - Reads the numpy files created from data_preperation.py
   - Splits them into train and test data, 80-20, in a balanced way, in terms of labels in train and test sets.
   - Creates a simple LSTM model with 128 units --> Batch Normalization --> Dense(64, activation="relu") --> Dense(3, activation="softmax")
   - Train it.
   
<br>

### <a href="https://github.com/Ggkenios/thesis/blob/main/plots.py">Plots</a> <br>
   - Creates a confusion matrix plot for all patients.

<br>
