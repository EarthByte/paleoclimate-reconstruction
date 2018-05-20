# Paleo-Climate Reconstruction
Paleo-Climate Precipitation Reconstruction using Bayesian Machine Learning

The main goal of the project is to model global precipitation using geological indicators (including location on continent and sedimentary deposits).

We have available to us the following datasets:
- non-regularly distributed climate-sensitive sedimentary deposits across geologic time (separated on scales of 10-20 million years) and their associated reconstructed geographical location

and the following simulated results:
- global rainfall modelled at 2.5 deg resolution during Miocene (Herold+ 2012)
- GPlates simulation of continental plates and topology across geologic time


## Step 0: Installation
* Check that you have necessary packages installed (see requirements.txt)
* download the query_paleogeography folder from: https://cloudstor.aarnet.edu.au/plus/s/HRkPptZuiZEm8OV 
* install pyGPlates (instructions at http://www.gplates.org/docs/pygplates/index.html )
* install GMT
  * Note that both the pyGPlates and query_paleogeography folders will need to be in the PYTHONPATH (the pyGPlates documentation talks about this).

## Step 1: Pre-processing raw data
Run
```sh 
python data_preprocessing.py
```
creates
* gridded input data maps in data/images/
* precipitation_gpmodel_data.csv in data/


## Step 2: Bayesian modelling
```sh
python run_GPplus.py
```
Note: This code is adapted from [GPplus](https://github.com/sebhaan/GPplus). Changes include:
* choice of functions and parameters depending on version of george being used
* references to new variable target_desc instead of Crime
* removal of dropout flag (was never used)
* getting and writing parameter-scaling terms (min and range) to csv-file
* adding exact prediction line to output plot


 