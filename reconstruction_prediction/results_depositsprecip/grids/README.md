# Grids of reconstructed prediction data
Each of the key result variables are provided as netCDF grids with the following attributes:

* Gridline node registration used [Geographic grid]
* Grid file format: nf = GMT netCDF format (32-bit float), COARDS, CF-1.5
* x_min: 0 x_max: 360 x_inc: 3 name: longitude [degrees_east] n_columns: 121
* y_min: -90 y_max: 90 y_inc: 3 name: latitude [degrees_north] n_rows: 61
* z_min: varies z_max: varies name: z
* Filename: era + time era + z varibale + .tif
* scale_factor: 1 add_offset: 0
* format: classic

Grids were generated from the era???results.csv data files using the Generic Mapping Tools-GMT 5.4.3. These commands were used to generate the grids:

```
for i in ../*.csv; 
	do base=${i%results.csv}; 
	nm=`echo "$base" | cut -c 7-`;
	echo $nm; 
	for j in {2..17}; 
		do var=$(awk -v taskID=$j '$1==taskID {print $2}' params.txt); 
		awk -F',' -v id=$j '{print $2, $1, $id}' $i > temp.xyz; 
		gmt blockmean -I3d -Rg temp.xyz > temp.block; 
		gmt nearneighbor -N2 -S500k -I3d -Rg temp.block -G${nm}Ma_${var}.nc; 
	done; 
done
```

These global files can be loaded in any of your favourite GIS tools and grid viewing applications (e.g. QGIS, Panoply, GMT, etc).
