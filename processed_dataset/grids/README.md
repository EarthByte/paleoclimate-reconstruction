# Grids of predictor data

Each of the key predictor variables are provided as nerCDF grids with the following attributes:

* Gridline node registration used [Geographic grid]
* Grid file format: nf = GMT netCDF format (32-bit float), COARDS, CF-1.5
* x_min: 0 x_max: 360 x_inc: 3 name: longitude [degrees_east] n_columns: 121
* y_min: -90 y_max: 90 y_inc: 3 name: latitude [degrees_north] n_rows: 61
* z_min: 0 z_max: 58.3100013733 name: z
* scale_factor: 1 add_offset: 0
* format: classic

Grids were generated from the predictor data csv files using the [Generic Mapping Tools-GMT](https://www.generic-mapping-tools.org/).
These commands were used to generate the grids:

```
for i in ../predictor_data*.csv 
do base=${i%.*}
	echo $base
	for j in 7..12
		do echo $j
		var=$(awk -v taskID=$j '$1==taskID {print $2}' params.txt)
		echo $var
		awk -F',' -v id=$j '{print $3, $2, $id}' $i > temp.xyz
		gmt blockmean -I3d -Rg temp.xyz > temp.block 
		gmt nearneighbor -N2 -S500k -I3d -Rg temp.block -G${base}_${var}.tif
	done
done
```

These global files can be loaded in any of your favourite GIS tools and grid viewing applications (e.g. QGIS, Panoply, GMT, etc).
