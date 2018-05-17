# paleoclimate-reconstruction
Paleo-Climate Precipitation Reconstruction using Bayesian Machine Learning


## Step 0: Installation
* Check that you have necessary packages installed (see requirements.txt)
* install query_paleogeography found at: https://cloudstor.aarnet.edu.au/plus/s/HRkPptZuiZEm8OV 
* install pyGPlates (instructions at http://www.gplates.org/docs/pygplates/index.html ) and GMT
  * Note that both pyGPlates and 'query_paleogeography' will need to be in the PYTHONPATH (the pyGPlates documentation talks about this).

## Step 1: Pre-processing raw data
Run 
>> python data_preprocessing.py