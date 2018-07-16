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
python data_preprocess_miocene.py
```
This code uses the Lithology database and precipitation simulation output in the data/ folder and creates:
* gridded input data maps in data/images/
* precipitation_gpmodel_data.csv in data/
Note: the plate reconstruction queried in this code is designed specifically for the Miocene and is consistent with the (Miocene) precipitation output. Therefore the output csv file is most appropriate for learning.


## Step 2: Bayesian modelling
```sh
cd GPplus2/
python run_GPplus.py
```
Note: This code is adapted from [GPplus](https://github.com/sebhaan/GPplus). Changes include:
* references to new variable target_desc instead of Crime
* writing covariance matrix and likelihood function manually (instead of using george)
* drawing predictions from multivariate normal
* getting and writing parameter-scaling terms (min and range) to csv-file
* two distinct GP length scale parameters (one for latitude, one for longitude)
* removal of dropout flag (was never used)


## Step 3: Prediction Input Data
Run
```sh 
python data_predictors_anytime.py
```
creates
* gridded input data maps in data/images/
* predictor_data_XXX_YYY.csv in data/
where 
* XXX is the each geological epoch listed in the deposit database between 6Ma and 251Ma
* YYY is either land or deposit. 


 