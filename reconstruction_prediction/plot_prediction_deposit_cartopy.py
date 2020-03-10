
# coding: utf-8

# # Plot Predicted Precipitation 
# 
# ## Madhura Killedar
# ## date   : 20/07/2018
# 

import sys
import os
import numpy as np
import pandas as pd
from math import asin, acos, sqrt, sin, cos, radians, degrees, asinh
from matplotlib import pyplot as plt
import matplotlib
import csv
import cartopy.crs as ccrs
matplotlib.use('Agg')

import seaborn as sns
print("Hello Prediction Plotting")




# USER CHOICES
if(len(sys.argv)!=4):
    sys.exit('Usage: python plot_prediction.py <input file> <one-word subject of folder where data input resides  and outputs will be kept>')

 
predict_folder = str(sys.argv[1])
predict_filename = str(sys.argv[2])
subject = str(sys.argv[3])

directory_plot =  predict_folder+'/'+subject

if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 

directory_plot =  predict_folder+'/'+subject + "/map_prediction"
 

if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 

directory_plot =  predict_folder+'/'+subject + "/map_prediction_uncert_"

if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 
 

directory_plot =  predict_folder+'/'+subject


print(predict_filename , ' predict_filename')

prediction_file = predict_folder +'/' + predict_filename  # example only

print("Prediction file path: %s"%(prediction_file))
 
  
#Read in the data  
predictions = pd.read_csv(prediction_file, header=None)
 
  
latitudes     = predictions.loc[:,0]
longitudes    = predictions.loc[:,1]


# ## SET UP GRIDS
lon_min = -180.
lon_max =  180.
lat_min =  -90.
lat_max =   90.
lon_spacing = 3 #2.51

lon_coords = np.arange(lon_min,lon_max-0.5*lon_spacing,lon_spacing) # left edge of grid
nlonbins = len(lon_coords)
print("Number of Longitudinal bins = %i"%nlonbins)
nlatbins = 2 * int(nlonbins * lat_max / 561.) # rough guess to allow a square bin at equator
print("Number of Latitudinal bins = %i"%nlatbins)
lat_coords = np.empty(nlatbins) # lower edge of grid
lat_spacing = np.empty(nlatbins)
for ilat in np.arange(nlatbins):
    coslat = 1. - 2.*ilat/float(nlatbins)
    lat_rad = acos(coslat)
    lat_coords[ilat] = degrees(lat_rad) - 90.
for ilat in np.arange(nlatbins-1):
    lat_spacing[ilat] = lat_coords[ilat+1] - lat_coords[ilat]
lat_spacing[nlatbins-1] = 90. - lat_coords[nlatbins-1]
print("Check central latitude bin-edges ...")
print(lat_coords[int(nlatbins/2-1):int(nlatbins/2+2)])
# remove polar grid-bins
nlatbins = nlatbins-2
lat_coords = lat_coords[1:-1]
lat_spacing = lat_spacing[1:-1]



# # PRECIPITATION
map_predict_actual = np.zeros((nlatbins,nlonbins))
map_predict_mean = np.zeros((nlatbins,nlonbins))
map_predict_low  = np.zeros((nlatbins,nlonbins))
map_predict_high = np.zeros((nlatbins,nlonbins))
map_predict_unct = np.zeros((nlatbins,nlonbins))
mask_exclude     = np.zeros((nlatbins,nlonbins))
list_mean = []
list_low  = []
list_high = []
list_unct = []
list_actual = []

if subject == 'coal':
    actual_col =  15
    mean_col  = 6 
    high_col =  12
    low_col =  9
    print(subject, ' is subject')

if subject == 'glacial':
    actual_col =   17
    mean_col  =  8
    high_col = 14
    low_col =  11
    print(subject, ' is subject')

if subject == 'evaporites':
    actual_col = 16
    mean_col  =  7  
    high_col =  13
    low_col =  10
    print(subject, ' is subject')


 
 

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
 
        here = (this_paleolongitude & this_paleolatitude)
 
        if(here.any()):
            map_predict_actual[nlatbins-ilat-1,ilon] = pred_wrapper(predictions.loc[here,actual_col].values)
            map_predict_mean[nlatbins-ilat-1,ilon] = pred_wrapper(predictions.loc[here,mean_col].values)
            map_predict_low[nlatbins-ilat-1,ilon]  = pred_wrapper(predictions.loc[here,low_col].values)
            map_predict_high[nlatbins-ilat-1,ilon] = pred_wrapper(predictions.loc[here,high_col].values)
            map_predict_unct[nlatbins-ilat-1,ilon] = pred_wrapper(map_predict_high[nlatbins-ilat-1,ilon] - map_predict_low[nlatbins-ilat-1,ilon])
            list_mean.append(map_predict_mean[nlatbins-ilat-1,ilon])
            list_low.append(map_predict_high[nlatbins-ilat-1,ilon])
            list_high.append(map_predict_low[nlatbins-ilat-1,ilon])
            list_unct.append(map_predict_unct[nlatbins-ilat-1,ilon])
            list_actual.append(map_predict_actual[nlatbins-ilat-1,ilon])
        else:
            mask_exclude[nlatbins-ilat-1,ilon] = True



# ## PLOT Prediction

#print( map_predict_mean)

xxx = 20


temp = predict_filename.split('.csv')

predict_filename = temp[0]
  

lons, lats = np.meshgrid(lon_coords,lat_coords)
actual_data = predictions[predictions.loc[:,actual_col]>0].loc[:,actual_col].values
actual_lat = predictions[predictions.loc[:,actual_col]>0].loc[:,0].values
actual_lon = predictions[predictions.loc[:,actual_col]>0].loc[:,1].values
map_predict_mean[map_predict_mean==0]=np.nan
map_predict_unct[map_predict_unct==0]=np.nan

# set up a map
fig = plt.figure(figsize=(16,12),dpi=150)
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
ax.gridlines(linewidth=0.1)

#Create the varible to plot
intensity = np.ma.masked_where(np.isnan(map_predict_mean), map_predict_mean) 
#Plot on map
mapscat=plt.pcolormesh(lons,-lats,intensity,shading='flat',cmap=plt.cm.gist_earth_r,transform=ccrs.PlateCarree())
#Add additional points
line=ax.scatter(actual_lon,actual_lat,c='darkorange',marker='d',transform=ccrs.PlateCarree(),label='Deposit locations') 
#Add the legend, colorbar and some text
plt.legend(loc=3,prop={'size': 18}) 
#Get the current time from the filename
time1=predict_filename.split('era')
time2=time1[1].split('results')
figtime=time2[0]
print("fig time=",figtime)
plt.text(14000000,7000000,str(figtime)+" Ma",size=20)

cbar=plt.colorbar(mapscat, ax=ax, orientation="vertical", pad=0.05, fraction=0.015, shrink=0.5)
cbar.set_label('Prediction probability of '+subject,labelpad=xxx,size=xxx)  
cbar.ax.tick_params(labelsize=xxx)


fig.savefig( directory_plot+"/map_prediction/"+ predict_filename+".pdf", pad_inches=0.6, bbox_inches='tight')
fig.clf()
 







# set up a map
fig = plt.figure(figsize=(16,12),dpi=150)
ax = plt.axes(projection=ccrs.Mollweide())
ax.set_global()
ax.gridlines(linewidth=0.1)

#Create the varible to plot
intensity = np.ma.masked_where(np.isnan(map_predict_unct), map_predict_unct) 
#Plot on map
mapscat=plt.pcolormesh(lons,-lats,intensity,shading='flat',cmap=plt.cm.gist_earth_r,transform=ccrs.PlateCarree())
#Add additional points
line=ax.scatter(actual_lon,actual_lat,transform=ccrs.PlateCarree(),label='Deposit location') 
#Add the legend, colorbar and some text
plt.legend(loc=3,prop={'size': 18}) 
#Get the current time from the filename
time1=predict_filename.split('era')
time2=time1[1].split('results')
figtime=time2[0]
print("fig time=",figtime)
plt.text(14000000,7000000,str(figtime)+" Ma",size=20)

cbar=plt.colorbar(mapscat, ax=ax, orientation="vertical", pad=0.05, fraction=0.015, shrink=0.5)
cbar.set_label('Uncertainty for '+subject,labelpad=xxx,size=xxx)
cbar.ax.tick_params(labelsize=xxx)


fig.savefig( directory_plot+"/map_prediction_uncert_/"+ predict_filename+".pdf", pad_inches=0.6, bbox_inches='tight')
fig.clf()
 
 