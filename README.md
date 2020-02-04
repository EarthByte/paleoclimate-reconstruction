# Paleo-Climate Reconstruction
Paleo-Climate Precipitation Reconstruction using Bayesian Machine Learning

The main goal of the project is to model global precipitation using geological indicators (including location on continent and sedimentary deposits).

We have available to us the following datasets:
- a database of non-regularly distributed climate-sensitive sedimentary deposits across geologic time (separated on scales of 10-20 million years) and their associated reconstructed geographical location `data/LithData_PaleoXY_Matthews2016_20180226.csv`
- a dictionary used to translate the lithology codes `data/LithologyCodes.csv`
- global rainfall modelled at 2.5 deg resolution during Miocene (Herold+ 2012) `data/PRECT_Average_annum.grd`
- GPlates simulation of continental plates and topology across geologic time (local installation)



## Step 0: Installation
* Check that you have necessary packages installed (see `requirements.txt`)
* download the `query_paleogeography/` folder from: https://cloudstor.aarnet.edu.au/plus/s/HRkPptZuiZEm8OV 
* install pyGPlates (instructions at http://www.gplates.org/docs/pygplates/index.html )
* install GMT
  * Note that both the pyGPlates and query_paleogeography folders will need to be in the PYTHONPATH (the pyGPlates documentation talks about this).

## Step 1: Pre-processing data for learning
Run
```
python data_preprocess_miocene.py
```
This should take about 4 minutes. The code uses the deposit database, lithology dictionary, and precipitation simulation output in the `data/` folder and creates:
* gridded input data maps in `data/images/`
* input data file: `data/learning_data_miocene_YYY.csv`

where 

* YYY is either `land` or `deposit` (corresponding to `data_subset` the user-set variable within the code)
  * for files with identifier `land`, the dataset will include all points on land (or shallow marine areas that contain deposits)
  * for files with identifier `deposit`, the dataset will include all points where some type of deposit has been found

_Note: these files will be created locally but are not to be pushed to the online repository_

The plate reconstruction queried in this code is designed specifically for the Miocene and is consistent with the (Miocene) precipitation output. Therefore the output csv file is most appropriate for learning.

 

## Step 2: Pre-process data for prediction
Run
```
python data_predictors_anytime.py
```
This code uses the deposit database and lithology dictionary in the `data/` folder and creates:
* gridded input data maps in `data/images/`
* input data files: `data/predictor_data_XXX_YYY.csv`

where 

* XXX is the each geological epoch listed in the deposit database between 6Ma (lower limit for the plate reconstruction query) and 251Ma (Permian)
* YYY is either `land` or `deposit`
  * for files with identifier `land`, the dataset will include all points on land (or shallow marine areas that contain deposits)
  * for files with identifier `deposit`, the dataset will include all points where some type of deposit has been found

_Note: these files will be created locally but are not to be pushed to the online repository_


## Step 3: Bayesian model: Predict missing deposit and estimate precip using Gaussian Process - Gibs Sampler based MCMC framework in Matlab


Note: This code is adapted from [GPplus](https://github.com/sebhaan/GPplus). Changes include:
* references to new variable target_desc instead of Crime 
* tba


* [Execute matlab code ](https://github.com/EarthByte/paleoclimate-reconstruction/blob/master/reconstruction_prediction/model/predmodel_framework.m)
* [Main results for all eras](https://github.com/EarthByte/paleoclimate-reconstruction/blob/master/reconstruction_prediction/model/results_all.csv)




## Step 4: Plot map of deposit and predictions
  
* gridded prediction and uncertainty maps  for deposits (coal, evaporites, glacial) and precip using python matplotlib
* [Run visualise shell script ](https://github.com/EarthByte/paleoclimate-reconstruction/blob/master/reconstruction_prediction/run_results_visualisations_.sh)

* [View maps coal estimation](https://github.com/EarthByte/paleoclimate-reconstruction/tree/master/reconstruction_prediction/results_depositsprecip/coal)

* [View maps evaporites estimation](https://github.com/EarthByte/paleoclimate-reconstruction/tree/master/reconstruction_prediction/results_depositsprecip/evaporites)

* [View maps glacial estimation](https://github.com/EarthByte/paleoclimate-reconstruction/tree/master/reconstruction_prediction/results_depositsprecip/glacial)


* [View maps precip estimation](https://github.com/EarthByte/paleoclimate-reconstruction/tree/master/reconstruction_prediction/results_depositsprecip/precitmap)

* use [Anaconda environment](environment.yml)



