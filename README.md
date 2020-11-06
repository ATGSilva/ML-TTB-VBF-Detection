# ML-TTB-VBF-Detection
2019/2020 MSci Particle Physics Machine-Learning Project

## Environment setup

```
conda create -c conda-forge --name MsciProject_TF python=3.7 uproot matplotlib numpy pandas tqdm opencv
pip install tensorflow
pip install hls4ml
```
N.B. perhaps could remove some explicit dependencies, as included when installing tensorflow/hls4ml?


## Read input training/testing data

This will read in root files containing 2D histograms (eta-phi axes, with bin contente being sum of pt of PUPPI candidates) and convert the histogram to numpy arrays.  There is one root file (containing one histogram) per event.

The input histograms are currently stored in my work area.  The output numpy arrays (one file per array/event) will appear in the directory 'TTBAR_numpy'.  The 'TTBAR' label is set by the 'Sample' variable in Reading_in_2dhist_uproot.py.  Changing 'Sample' to 'SNU' will produce the numpy arrays for the single neutrino inputs (i.e. background).

```
cd TestTrainDataCreation
python Reading_in_2dhist_uproot.py 
```

## Convert the numpy arrays to pickle format

Reads in the numpy arrays for TTBAR and SNU from previous steps, splits these into a training and testing set.  The two training and testing sets for signal/background are combined and saved to 'test_<training/validation>_<x/y>.pkl' files.

```
python Pickle_creator_two_interactions.py 
```
A similar script "Pickle_creator_three_interactions.py" exists, but I have not touched it.  I belive it's for multiclassifier models e.g. for distinguishing ttbar vs VBF vs single neutrino.

## Create and train a tensorflow model

Define, train, and evaluate a model using keras in tensorflow.

```
cd ../TFModelCreation
python TTB_detector_implemented.py 
```
