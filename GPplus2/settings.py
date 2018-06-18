"""
Define directories and options, here as an example for modeling crime offences.
Input file requirements: Input csv file must include one column labeled "region_id" which holds the ID number for each location
and two columns "centroid_x" and "centroid_y" which hold the x and y coordinate for each location, respectively.

If shapefiles need to be processed, the requirement are two input files, one containing the numbers for the the model feature data
(e.g. demographic data) for each region, and another file that contains the shape file for each region. 
Both files are linked via same ID's named as "region_id".
"""

# Main input directory:
data_path = "../data/"
# Filename for main input csv file:
fname_input = "precipitation_gpmodel_data.csv" # 
# Polygon shapefiles of areas boundaries, only needed if coordinates not already calculated and in main input file:
shape_filename = "foo" # assumes in data_path directory, optional.
shape_region = 'bar' # set to None if no particular region exctracted and all shapedata should be included, optional.
# If center coordinates already calculated, provide data in main file with two column names "centroid_x" and "centroid_y":
# Filename for crime input data including crime numbers and demographic/environment features
# main output directory
outpath = "../GPresults/"
# specific output directory for result plots, maps and tables
outmcmc = outpath # add any subdirectory if required

####### Define options for analysis and visualisation

split_traintest = 0.2 # splits data in train and test data, provide fraction of test
simulate    = False # Creates simulated data with 2dim spatial coordinates and 2 features and then runs code on this data.
calc_center = False # Calculates centroids from polygons and converts from Lat/Lng to cartesian with center coord below.
                   # If calc_center = False, coordinates have to be provided in main input file [fname_input]
                   # using the two column names "centroid_x" and "centroid_y"
center_lat = 0.  # Latitude coordinate for spatial center for conversion into cartesian coordinates, e.g. NSW centroid
center_lng = 0.  # Longitude coordinate for spatial center, e.g. NSW centroid
create_maps = False # creates interactive html maps of crime results, not included for simulated data.
# GP setup
kernelname = 'expsquared' # choose from: 'expsquared', 'matern32' or 'rationalq' (Rational Quadratic)
# MCMC setup; see emcee documentation for more details (http://dfm.io/emcee/current/)
nwalkers = 24 # Number of walkers, recommended at least 2 * Nparameter + 1 (recommended more)
niter = 2500  # Number of iterations, recommended at least 500
nburn = 1500 # Number of iterations, recommended at least 20% of niter, check sampler chains for convergence.
nfold_cross = 1 # Number of x-fold for cross-validation for test-train sets; set to 1 for only one run 
#(Notes: test first with nfold-cross = 1; computational time ~ nfold_cross; also not all plotting availabable yet for nfold_cross>1)
use_log = False # select false if input features need NOT to be converted into log-space.
# Note that all input features will be normalized from 0 to 1, whether log-space enabled or not. 


###### List of input features

# use population number by area of region instead of just absolute population numbers?
#pop_per_area = True

target_name = 'Log Precipitation'
target_desc = 'Log Precipitation' # for plot axes labels

# Identify features in data that should be used for linear regression
# Names must match column names in header, replace names below accordingly:
x_feature_names = ['Coal Deposits',
                   'Evaporites Deposits',
                   'Glacial Deposits',
                   #'Elevation',
                   'Sqrt Elevation',
                   #'Dist to Shore'
                   'Sqrt Dist',
                   #'ASinh Elevation Sc50',
                   #'Sin Angle Shore',
                   #'Cos Angle Shore'
                   ]


# Optionally provide additional description names for features, replace names accordingly:
x_feature_desc = ['Coal Deposits',
                  'Evaporites Deposits',
                  'Glacial Deposits',
                  #'Elevation',
                  'Sqrt Elevation',
                  #'Distance to Shore'
                  'Sqrt Distance',
                  #'ASinh Elevation Sc50',
                  #'Eastern  shore (sin)',
                  #'Northern shore (cos)'
                  ]
                  

#######################################
