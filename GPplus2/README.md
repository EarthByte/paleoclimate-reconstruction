# GPplus
GPplus combines a parametric model for the relationship between the target feature and location specific characteristics and an additive nonparametric model for spatial dependencies using Gaussian Processes (GP). The combination of Bayesian linear regression and spatial dependencies is extremely flexible and applicable to a large suite of data science problems, as tested in social-demographic, crime, environment, and geophysical analysis. 

Inference about model parameters and their uncertainty is carried out via a fast parallel Markov chain Monte Carlo (MCMC) sampler, which is a very efficient way for multidimensional integration. For sampling the posterior distribution, GPplus uses the affine-invariante MCMC ensemple sampler, "emcee" (Foreman-Mackey et al, 2013, Goodman and Weare, 2010).

The implemented method is a fully probabilistic approach, allowing uncertainties in prediction and inference to be quantified via the posterior distributions of interest. By using Bayesian updating, these predictions and inferences are dynamic in the sense that they change as new information becomes available. 

GPplus provides mutliple plots for evaluation and inference. The results, parameters, and uncertainties are stored as csv files and the complete posterior distribution as npy files. If required, more postprocessing and visualisation scripts can be requested from author (sebastian.haan@sydney.edu.au).

Multiple options for processing are included (see settings.py):
 * Option for generating simulated data.
 * Option for extracting spatial component from polygon shapefiles.
 * Options for MCMC setup: number of iterations, number of walkers, length of burn-in phase
 * Options for multiple kernels to model spatial cross-correlations
 * Option for generating html maps for result visualisation.
 * Options for splitting in train and test sets and x-fold cross-validation

GPplus is being actively developed in [a public repository on GitHub](https://github.com/sebhaan/GPplus). If you have any questions, please [open an issue](https://github.com/sebhaan/GPplus/issues).

Additional dependencies
---------------------

* emcee, see [here](https://github.com/dfm/emcee.git) for installation from Github
* george, see [here](https://github.com/dfm/george.git) for installation from Github (only for this GPplus version, will be replaced with our own in-house GP in next versions)
* geopandas (optional, for geospatial pre-processing of polygon shapefiles)
* geopy (optional, for faster geodesic distance calculation)
* folium (optional, for intearctive html mapping)

License & Attribution
---------------------

Copyright 2018 Sebastian Haan and contributors.

The source code is made available under the terms of the MIT license.

If you make use of this code, please acknowledge GPplus and authors.
