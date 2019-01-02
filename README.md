# dreem-challenge (OBT Machine Learning Class 18-19)

Sleep Stage Classification Kaggle Challenge.

## What is Dreem headband ? 

Dreem headband allows doing polysomnography at home signal thanks to three kinds of sensors: electroencephalogram (EEG), pulse oximeter and accelerometer signals.

## What is project about ? 

Since the Dreem headband records a lot of nights every day, the aim is to developp the most accurate automatic sleep staging algorithms. The idea is to develop an algorithm of sleep staging able to differentiate between Wake, N1, N2, N3 and REM on windows of 30 seconds of raw data. The raw data includes 7 eegs channels in frontal and occipital position, 1 pulse oximeter infrared channel, and 3 accelerometers channels (x, y and z).

## Content 

- Classical Machine Learning Approach 
  - Signal processing of the EEG and accelerometer signals (band-pass filtration)
  - Extraction of 267 features on raw and filtered signals
  - Random Forest on extracted features

- Deep Learning Approach
  - Stacking the raw EEG signals as an input data 
  - Running an 11-layers CNN (convolutional neural network) 
