"""
This is the main script for runnig a joined model of bayesian linear regression 
with a spatial gaussian process prior. The posterior distribution is sampled with 
"emcee", an affine-invariant ensemble MCMC sampler (Foreman-Mackey et al, 2013;
Goodman and Weare, 2010). The results, parameters and uncertainties are stored as csv files. 
Multiple plots are generated for evaluation (please contact author if more evaluation plots requiered)  

This machine-learning implementation is a fully probabilistic approach to modelling spatial dependent data 
while reflecting all uncertainties in the prediction as well as the uncertainties surrounding model parameters.
The framework has been developed for multiple use-cases and has been first tested by modelling. 
the dependency between crime data and environmental factors such as demographic characteristics and spatial location. 
(Marchant et al. 2018).

Currently multiple options are included (see settings.py): 
 - Option for generating simulated data.
 - Option for extracting spatial component from polygon shapefiles.
 - Options for multiple kernels.
 - Option for generating html maps for result visualisation.
 - options x-fold cross-validation.
 - options for splitting in train and test sets.
 
Version 0.1
Author: Seb Haan
12 Feb 2018
"""

from __future__ import division, print_function
import os
from geo_spatial import Shape # handling shapefile and geometry
from gpmc import GPMC # handling main MCMC modeling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from settings import *


"""
Script to run main analysis:
1) Read in crime data and spatial coordinates of areas and process geometries into cartesian grid
to calculate centroids, areas, distances etc
Note that both input file, crime and spatial, require a column labeled with "region_id" for merging both datasets.
"""

# initialize mcmc:
gpmc = GPMC(outmcmc = outmcmc, split_traintest = split_traintest)
# Create simulated data if enabled:
if simulate:
    print("Creating simulated data ...")
    gpmc.create_data(outpath, x_min=0., x_max=10, nsize=20, noise=0.1)
    fname_input = "simdata2D.csv"
    x_feature_names = ["x1", "x2"]
    x_feature_desc = ["Feature x1", "Feature x2"]
    calc_center = False
    pop_per_area = False
    create_maps = False
# read in spatial data and inigialize classes:
spatial = Shape(data_path, shape_filename, outpath, calc_center)
if calc_center and not simulate:
    print("Creating geometry from shapefiles ...")
    spatial.create_geometry(center_lat, center_lng, region = shape_region) # center coorinates are for NSW here


# 2) Read in target data (e.g. crime offences) and feature data (e.g. demographics from census data) and combine with spatial data
print("Reading in data and combining with spatial geometry ...")
if simulate:
    spatial.read_csvdata(outpath + fname_input)
else:
    spatial.read_csvdata(data_path + fname_input)
if calc_center:
    spatial.combine_data()
else:
    spatial.data_comb = spatial.data_df # centroid_x and centroid_y already in csv file

# 3) Run main MCMC algorithm to sample over linear gression coefficients, noise, spatial GP hyperparameters etc:
print("Running MCMC sampler on spatial plus feature data")

# make parameter list description
GP_par_names = ["GP1", "GP2", "GP3"]
par_list = np.concatenate((["Alpha", "Sigma"], x_feature_desc, GP_par_names))

# Store numpy array of GP coordinates
x_gp = np.vstack((spatial.data_comb.centroid_x.values,spatial.data_comb.centroid_y.values)).T

# if pop_per_area:
#     x_feature_desc[-1] = x_feature_desc[-1] + " per Area"
#     spatial.data_comb.Tot_persons_C11_P = spatial.data_comb.Tot_persons_C11_P.values/ spatial.data_comb.shape_area_sqkm.values

# Read in X-features into numpy array
x_in = spatial.data_comb[x_feature_names[0]].values
for i in range(len(x_feature_names)-1):
    x_in = np.vstack((x_in, spatial.data_comb[x_feature_names[i+1]].values))
x_in = x_in.T
gpmc.demog = x_in

# Convert X-features into log space and Normalize
if use_log:
    x_log = np.log(x_in)  # make nat logarithmic
    x_norm, x_norm_min, x_norm_range = gpmc.norm_data(x_log) # normalize each feature from 0 to 1
else:
    x_norm, x_norm_min, x_norm_range = gpmc.norm_data(x_in)
x_gp_norm, x_gp_norm_min, x_gp_norm_range  = gpmc.norm_data(x_gp)

# Read in target data (vector):
y = spatial.data_comb[target_name] 

# Save all processed data for MCMC
gpmc.out_data(y, x_gp_norm, x_norm)

# Splits set in train and test set.
idx = spatial.data_comb.region_id
x_norm_train, x_blr_test, x_gp_norm_train, x_gp_test, y_train, y_test, idx_train, idx_test =\
    train_test_split(x_norm, x_gp_norm, y, idx, test_size = split_traintest, random_state = 0)

# Loads train and test dataset into MCMC class for processing:
gpmc.load_data(y_train, x_gp_norm_train, x_norm_train, y_test, x_gp_test, x_blr_test)

# Run actual MCMC:
gpmc.calc_mcmc(nwalkers, niter, nburn, split_traintest)


# 4) make summary calculations for MCMC performance evaluation
gpmc.calc_residual()  # print residuals
gpmc.calc_samples(nitersample=30)
gpmc.results_file.close() # closes output text file (automatically opened in gpmc initialization)

# 5) Plot result histograms and diagrams
gpmc.plot_hist(par_list)
gpmc.plot_niter(par_list)
gpmc.plot_corner(par_list)
gpmc.plot_diagr(x_feature_desc)

# 6) save parameter stats as csv
gpmc.create_param_csv(par_list)
gpmc.create_scaling_csv(x_feature_names, x_norm_min, x_norm_range, 'BLR')
gpmc.create_scaling_csv(['X coords','Y coords'], x_gp_norm_min, x_gp_norm_range, 'GP')

# 7) Create maps, can take some time and disk space.
if create_maps and not simulate and (nfold_cross == 1):
    idx_comb = np.hstack((idx_train.values,idx_test.values))
    train = np.ones((len(idx_train)))
    test = np.zeros((len(idx_test)))
    train_comb = np.hstack((train,test)) # ones for train data, zeros for test data
    y_model = np.hstack((gpmc.y_model,gpmc.y_model_test))
    mu_blr = np.hstack((gpmc.mu_blr, gpmc.mu_blr_test))
    residual = np.hstack((gpmc.residual,gpmc.residual_test))
    residual_blr = np.hstack((gpmc.residual_blr,gpmc.residual_blr_test))
    mu_gp = np.hstack((gpmc.mu_gp,gpmc.mu_gp_test))
    head = ['region_id','train_mcmc','log_crime_model_mcmc', 'log_residual_mcmc','log_residual_blr_mcmc', 'log_blr_model', 'log_gp_model']
    comb = pd.DataFrame(data = np.array([idx_comb, train_comb, y_model, residual, residual_blr, mu_blr, mu_gp]).T, columns = head)
    comb.region_id = comb.region_id.values.astype(int)
    spatial.data_comb = spatial.data_comb.merge(comb, how='inner', on=['region_id'])
    df_csv2 = spatial.data_comb.drop('geometry', axis=1)
    df_csv2.to_csv(outmcmc + 'mcmc_results.csv')
    # Create map for each feature:
    spatial.create_maps2(outmcmc, 'log_crime_model_mcmc', center_lat, center_lng)
    spatial.create_maps2(outmcmc, 'log_residual_mcmc', center_lat, center_lng)
    spatial.create_maps2(outmcmc, 'log_gp_model', center_lat, center_lng)

plt.close("all")
print("GPplus FINISHED")

