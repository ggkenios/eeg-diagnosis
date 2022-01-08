&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; <img src="https://i.imgur.com/UwbMboU.png" width="190" height="200">

# &emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; EEG Diagnosis

## <a href="https://github.com/ggkenios/eeg-diagnosis/tree/main/common">Common</a> directory <br>
* <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/constants.py">Constants</a> <br>
   - Set paths, model, hyperparameters, and other needed constants.
<br>

* <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/functions">Functions</a> directory<br>
   - <b><a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/functions/model.py">Model</a></b> related functions, like build, compile and checkpoint ones
   - <b><a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/functions/data_processing.py">Data processing</a></b> functions, mainly for patient split and tesnor preperation
   - <b><a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/functions/plots.py">Plot</a></b> functions, like train-val accuracy/loss, and confusion matrices
   - <b><a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/common/functions/voting.py">Voting</a></b> ensemble functions, both hard and soft, to make decisions per patient.
<br>
   
## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/data_preprocessing.py">Data Preperation</a> <br>
   * Reading data iteratively <br>
   * Applies band pass filter <br>
   * Applies notch filter <br>
   * Applies Fast Fourier Transform <br>
   * Stores data in 3 numpy files: <br>
     - x: Array of shape (-1, 1000, 19) for patient's data. So, basically a list of 2d arrays: 1000 datapoints (2 seconds-segment) for 19 channels. <br>
     - y: Array of shape (-1) that tracks the labels for each 2-second data segment. <br>
     - z: Array of shape (-1) that tracks the patient ID for each 2-second data segment. <br>

<br>

## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/split_by_segment">Split by segment</a> directory <br>
In this approach we split data into training-test sets by segments. That means segments from a patient exist in both train and test sets. <br>
The directory contains 2 modules. One to run an <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/split_by_segment/split_80_20.py">80-20 split</a> and the other a <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/split_by_segment/split_5_fold.py">5-fold cross-validation</a> one. <br>
   - Reads the numpy files created from data_preperation.py <br>
   - Splits them into train and test data, in a balanced way, in terms of labels in train and test sets. <br>
   - Reads the model from model.py <br>
   - Trains it. <br>
   - Saves weights. <br>

<br>

## <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/split_by_patient">Split by patient</a> directory <br>
In this approach we split data into training-test sets by patient. That means segments from a patient can only exist in either train or test set. <br>
The file contains 2 modules. One to run an <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/split_by_patient/split_45_9.py">45-9 split</a> and the other a <a href="https://github.com/ggkenios/eeg-diagnosis/blob/main/split_by_patient/split_6_fold.py">5-fold cross-validation</a> one. <br>
   - Reads the numpy files created from data_preperation.py <br>
   - Splits them into train and test data, 80-20, by patient, in a balanced way. That means that segments of the same patient cannot exist on both train and test set. <br>
   - Reads the model from model.py <br>
   - Trains it. <br>
   - Saves weights. <br>

<br>

