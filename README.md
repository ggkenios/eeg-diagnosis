&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; <img src="https://i.imgur.com/UwbMboU.png" width="190" height="200">

# &emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; Thesis

## <a href="https://github.com/ggkenios/eeg-diagnosis/tree/main/common">Common Folder</a> <br>
* <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/constants.py">Constants</a> <br>
   - Set paths, model, hyperparameters, and other needed constants.
<br>

* <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/support_functions.py">Support Functions</a> <br>
   - Functions to: Create model, compile model, calculate majority vote
<br>

* <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/patient_validation.py">Patient Validation</a> <br>
   - Support code, for <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/plots.py">plots.py</a> to calculate majority vote, after the model is trained
<br>
<br>

## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/data_preperation.py">Data Preperation</a> <br>
   * Reading data ilteratively and stores them in 3 numpy files: <br>
     - x: Array of shape (-1, 1000, 19) for patient's data. So, basically a list of 2d arrays: 1000 datapoints (2 seconds-segment) for 19 channels. <br>
     - y: Array of shape (-1) that tracks the labels for each 2-second data segment. <br>
     - z: Array of shape (-1) that tracks the patient ID for each 2-second data segment. <br>

<br>

## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/model_train.py">Model Train</a> <br>
   - Reads the numpy files created from data_preperation.py <br>
   - Splits them into train and test data, 80-20, in a balanced way, in terms of labels in train and test sets. <br>
   - Read the model from support_functions.py <br>
   - Train it. <br>
   
<br>

## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/model_train_by_patient.py">Model Train by patient</a> <br>
   - Reads the numpy files created from data_preperation.py <br>
   - Splits them into train and test data, 80-20, by patient, in a balanced way. That means that segments of the same patient cannot exist on both train and test set. <br>
   - Read the model from support_functions.py <br>
   - Train it. <br>
   
<br>

## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/plots.py">Plots</a> <br>
   - Creates a confusion matrix plot for all patients using hard majority vote on each one's segments.

<br>
