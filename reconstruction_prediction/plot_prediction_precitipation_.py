
# coding: utf-8

 

from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
from math import asin, acos, sqrt, sin, cos, radians, degrees, asinh
from matplotlib import pyplot as plt
import seaborn as sns
import csv


from matplotlib import rc
font = {'family': 'serif',
        'serif' : ['Palatino'],
        'weight': 'bold',
        'size'  : 20}
rc('font', **font)
rc('text', usetex=True)
from matplotlib import rcParams
fig_dpi = 150
rcParams['figure.dpi']= fig_dpi

print("   Begin *******                   ********           *********  ")



# ## Input/Output Folders



# USER CHOICES
if(len(sys.argv)!=4):
    sys.exit('Usage:  input problem no. samples.   see run.sh')

 

predict_folder = str(sys.argv[1])
 


predict_filename = str(sys.argv[2])



directory_plot =  predict_folder+"/precitmap"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 



prediction_filepath = predict_folder +'/' + predict_filename  # example only

print("Prediction file path: %s"%(prediction_filepath))


print("Prediction predict_filename xx: %s"%(predict_filename))
  


directory_plot =  predict_folder+"/precitmap/map_prediction"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 



directory_plot =  predict_folder+"/precitmap/plot_graph"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 



directory_plot =  predict_folder+"/precitmap/map_actual"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 


directory_plot =  predict_folder+"/precitmap/snapshot_plot"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 


directory_plot =  predict_folder+"/precitmap/map_prediction_uncert"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 


directory_plot =  predict_folder+"/precitmap/map_prediction_diff"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 

type_pred = str(sys.argv[3])
 

directory_plot =  predict_folder+"/precitmap"
 

 

subject = 'prediction'
 

# read file
predictions = pd.read_csv(prediction_filepath, header=None)

#print(predictions)

 
  
latitudes     = predictions.loc[:,0]
longitudes    = predictions.loc[:,1]
 



# ## SET UP GRIDS
lon_min = -180.
lon_max =  180.
lat_min =  -90.
lat_max =   90.
lon_spacing = 3. #2.51

lon_coords = np.arange(lon_min,lon_max-0.5*lon_spacing,lon_spacing) # left edge of grid
nlonbins = len(lon_coords)
#print("Number of Longitudinal bins = %i"%nlonbins)
nlatbins = 2 * int(nlonbins * lat_max / 561.) # rough guess to allow a square bin at equator
#print("Number of Latitudinal bins = %i"%nlatbins)
lat_coords = np.empty(nlatbins) # lower edge of grid
lat_spacing = np.empty(nlatbins)
for ilat in np.arange(nlatbins):
    coslat = 1. - 2.*ilat/float(nlatbins)
    lat_rad = acos(coslat)
    lat_coords[ilat] = degrees(lat_rad) - 90.
for ilat in np.arange(nlatbins-1):
    lat_spacing[ilat] = lat_coords[ilat+1] - lat_coords[ilat]
lat_spacing[nlatbins-1] = 90. - lat_coords[nlatbins-1]
#print("Check central latitude bin-edges ...")
#print(lat_coords[nlatbins/2-1:nlatbins/2+2])
# remove polar grid-bins
nlatbins = nlatbins-2
lat_coords = lat_coords[1:-1]
lat_spacing = lat_spacing[1:-1]




# # PRECIPITATION
map_predict_mean = np.zeros((nlatbins,nlonbins))

map_predict_actual = np.zeros((nlatbins,nlonbins))
map_predict_low  = np.zeros((nlatbins,nlonbins))
map_predict_high = np.zeros((nlatbins,nlonbins))
map_predict_unct = np.zeros((nlatbins,nlonbins))
map_predict_diff = np.zeros((nlatbins,nlonbins))
mask_exclude     = np.zeros((nlatbins,nlonbins))

list_mean = []
list_low  = []
list_high = []
list_unct = [] 
list_actual = [] 
list_diff = []

 
# what happens if there are (fixed by David Kohn)
# multiple longitudes and latitudes
# within the specified grid area?
# we take the mean of the predictions
# for all the latitudes and longitudes
# that are within that grid
take_pred_mean = True

# pred_wrapper is the wrapper put around predicted values
# i.e. if multiple values, take the mean
pred_wrapper = lambda x: x
if take_pred_mean: 
    pred_wrapper = lambda x: np.mean(x)

 

for ilon,lon_low in enumerate(lon_coords):
    lon_high = lon_low + lon_spacing
    for ilat,lat_low in enumerate(lat_coords):
        lat_high = lat_low + lat_spacing[ilat]
        this_paleolongitude = (lon_low<longitudes) & (longitudes<lon_high)
        this_paleolatitude  = (lat_low<latitudes) & (latitudes<lat_high)
        here = this_paleolongitude & this_paleolatitude



        if(here.any()):


            map_predict_actual[nlatbins-ilat-1,ilon] =  pred_wrapper(predictions.loc[here,2] )
            map_predict_mean[nlatbins-ilat-1,ilon] =  pred_wrapper(predictions.loc[here,2] )
            map_predict_low[nlatbins-ilat-1,ilon]  =  pred_wrapper(predictions.loc[here,3] )
            map_predict_high[nlatbins-ilat-1,ilon] =  pred_wrapper(predictions.loc[here,4] )
            map_predict_unct[nlatbins-ilat-1,ilon] = map_predict_high[nlatbins-ilat-1,ilon] - map_predict_low[nlatbins-ilat-1,ilon] 
            map_predict_diff[nlatbins-ilat-1,ilon] = map_predict_mean[nlatbins-ilat-1,ilon] -map_predict_actual[nlatbins-ilat-1,ilon]

            list_mean.append(map_predict_mean[nlatbins-ilat-1,ilon])
            list_low.append(map_predict_high[nlatbins-ilat-1,ilon])
            list_high.append(map_predict_low[nlatbins-ilat-1,ilon])
            list_unct.append(map_predict_unct[nlatbins-ilat-1,ilon])  
            list_diff.append(map_predict_diff[nlatbins-ilat-1,ilon])
            list_actual.append(map_predict_actual[nlatbins-ilat-1,ilon])

        else:
            mask_exclude[nlatbins-ilat-1,ilon] = True



# ## PLOT Prediction

#print(list_actual)


print("minimum mean prediction is "+str(np.ma.array(list(map_predict_mean),mask=mask_exclude).min()))
print("maximum mean prediction is "+str(np.ma.array(list(map_predict_mean),mask=mask_exclude).max()))
fig, ax_prec_pred = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.9, as_cmap=True)
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_pred.set_title('Prediction for '+subject)
sns.heatmap(map_predict_mean, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_pred, cbar_ax=cbar_ax)
ax_prec_pred.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_pred.set_ylabel('Paleolatitude',  labelpad=10)
#fig.savefig(predict_folder+"map_prediction_"+subject+".png", pad_inches=0.6) 
fig.savefig(directory_plot+"/map_prediction/"+subject+predict_filename+".png", pad_inches=0.6)

fig.clf() 

x = np.arange(0, len(list_actual), 1); 
plt.plot(x, list_mean, label='pred. (mean)')
plt.plot(x, list_low, label='pred.(5th percen.)')
plt.plot(x, list_high, label='pred.(95th percen.)')
plt.fill_between(x, list_low, list_high, facecolor='g', alpha=0.4)
plt.legend(loc='upper right')

plt.title("Prediction with uncertainty ")

plt.ylabel('Precitipation')

plt.xlabel('Grid indentification number')
plt.savefig(directory_plot+"/plot_graph/"+subject+predict_filename+".png")
plt.clf()

 
plt.plot(x[0:100], list_mean[0:100], label='prediction')
plt.plot(x[0:100], list_low[0:100], label='pred.(5th percen.)')
plt.plot(x[0:100], list_high[0:100], label='pred.(95th percen.)')
plt.fill_between(x[0:100], list_low[0:100], list_high[0:100], facecolor='g', alpha=0.4)
plt.legend(loc='upper right')

plt.title("Prediction with uncertainty ")

plt.xlabel('Grid indentification number')
plt.ylabel('Precitipation')
plt.savefig(directory_plot+"/snapshot_plot/"+subject+".png")
plt.clf()
 

fig, ax_prec_unct = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0., rot=0.4, as_cmap=True)
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_unct.set_title('Uncertainty in Prediction for '+subject)
sns.heatmap(map_predict_unct, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_unct, cbar_ax=cbar_ax)
ax_prec_unct.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_unct.set_ylabel('Paleolatitude',  labelpad=10)
fig.savefig(directory_plot+"/map_prediction_uncert/"+subject+predict_filename+".png", pad_inches=0.6)

fig.clf() 



if (type_pred == "miocene") or (type_pred == "eocene"):

	fig, ax_prec_pred = plt.subplots(figsize=(18,6))
	fig.tight_layout()
	fig.subplots_adjust(right=0.9)
	cmap = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.9, as_cmap=True)
	cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
	ax_prec_pred.set_title('Actual values -  '+subject)
	sns.heatmap(map_predict_actual, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_pred, cbar_ax=cbar_ax)
	ax_prec_pred.set_xlabel('Paleolongitude', labelpad=10)
	ax_prec_pred.set_ylabel('Paleolatitude',  labelpad=10)
	fig.savefig(directory_plot+"/map_actual/"+subject+predict_filename+".png", pad_inches=0.6)
	fig.clf() 


	print("minimum uncertainty is "+str(np.ma.array(list(map_predict_diff),mask=mask_exclude).min()))
	print("maximum uncertainty is "+str(np.ma.array(list(map_predict_diff),mask=mask_exclude).max()))
	fig, ax_prec_unct = plt.subplots(figsize=(18,6))
	fig.tight_layout()
	fig.subplots_adjust(right=0.9)
	cmap = sns.cubehelix_palette(8, start=0., rot=0.4, as_cmap=True)
	cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
	ax_prec_unct.set_title('Residual (difference) in Prediction for '+subject)
	sns.heatmap(map_predict_unct, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_unct, cbar_ax=cbar_ax)
	ax_prec_unct.set_xlabel('Paleolongitude', labelpad=10)
	ax_prec_unct.set_ylabel('Paleolatitude',  labelpad=10)
	fig.savefig(directory_plot+"/map_prediction_diff/"+subject+predict_filename+".png", pad_inches=0.6)
	fig.clf()

	'''plt.plot(x, list_actual, label='actual')
	plt.plot(x, list_mean, label='pred. (mean)')
	plt.plot(x, list_low, label='pred.(5th percen.)')
	plt.plot(x, list_high, label='pred.(95th percen.)')
	plt.fill_between(x, list_low, list_high, facecolor='g', alpha=0.4)
	plt.legend(loc='upper right')

	plt.title("Prediction with uncertainty ")

	plt.ylabel('Precitipation')

	plt.xlabel('Grid indentification number')
	plt.savefig(directory_plot+"/plot/"+subject+predict_filename+".png")
	plt.clf()'''


	plt.plot(x[0:100], list_actual[0:100], label='actual')
	plt.plot(x[0:100], list_mean[0:100], label='prediction')
	plt.plot(x[0:100], list_low[0:100], label='pred.(5th percen.)')
	plt.plot(x[0:100], list_high[0:100], label='pred.(95th percen.)')
	plt.fill_between(x[0:100], list_low[0:100], list_high[0:100], facecolor='g', alpha=0.4)
	plt.legend(loc='upper right')

	plt.title("Prediction with uncertainty ")

	plt.xlabel('Grid indentification number')
	plt.ylabel('Precitipation')
	plt.savefig(directory_plot+"/snapshot_plot/"+subject+predict_filename+".png")
	plt.clf()

	print("minimum uncertainty is "+str(np.ma.array(list(map_predict_unct),mask=mask_exclude).min()))
	print("maximum uncertainty is "+str(np.ma.array(list(map_predict_unct),mask=mask_exclude).max()))

	fig, ax_prec_unct = plt.subplots(figsize=(18,6))
	fig.tight_layout()
	fig.subplots_adjust(right=0.9)
	cmap = sns.cubehelix_palette(8, start=0., rot=0.4, as_cmap=True)
	cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
	ax_prec_unct.set_title('Uncertainty in Prediction for '+subject)
	sns.heatmap(map_predict_unct, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_unct, cbar_ax=cbar_ax)
	ax_prec_unct.set_xlabel('Paleolongitude', labelpad=10)
	ax_prec_unct.set_ylabel('Paleolatitude',  labelpad=10)
	fig.savefig(directory_plot+"/map_prediction_uncert/"+subject+predict_filename+".png", pad_inches=0.6)

	fig.clf() 
		 




	 



	

