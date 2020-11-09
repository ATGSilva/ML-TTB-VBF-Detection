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

## Implement model in HLS

```
cd ../hls4ml/TTB_Detection/
hls4ml convert -c hls4ml_config.yml
```
This fails for me at the moment, with error:

```
File "<somepath>/hls4ml/converters/keras_to_hls.py", line 184, in keras_to_hls
    input_layer['input_shape'] = layer_config[0]['config']['batch_input_shape'][1:]
KeyError: 'batch_input_shape'
```

## Reamining steps, not yet tested

These are dumped as-is from the previous students.

        hls4ml build -p [proj] -a
 
proj should be the output from the previous command.
 
This should produce a working Vivado HLS project which you can now transfer to Excession or similar.
 
Synthesize and export in Vivado HLS. You can use CSim to simulate behaviour at this stage. We did this by using two bits of code, you will need to greatly adapt these. The first converts test data to CSV format: https://github.com/AminWetzel-Shirley/ML-TTB-VBF-Detection/blob/master/Analysis/Csv%20generation.py
 
The dimensions need to be configured within the above script. If you stray from the Pickle format or save your data within Pickles differently to our method you will need to change this script slightly. But the idea is to generate the test data into a data type that C++ can access.
 
The second script you will have to make and goes in your Vivado HLS project to run the CSim and access the data generated in the previous step: https://github.com/AminWetzel-Shirley/ML-TTB-VBF-Detection/blob/master/Analysis/test_bench_addon
 
The above code is integrated into code that Vivado HLS generates for performing CSim. You will have to adapt this for your purposes.
 
Finally, synthesise and implement using Vivado.
 
