
#!/bin/sh 
echo Running all 	


## declare an array variable 
 

declare -a list=("miocene"  "eocene" "eocene" "other" "other" "other" "other" "other" "other" "other" "other" "other" "other" "other" )


python split_eras.py 


x=0


declare -a eras=("14" "38" "39" "40" "51" "61" "77" "101" "129" "154" "182" "219" "242" "249")

 

#Note 40Ma refers to 28Ma. 39Ma refers to 4PIC 38Ma model

 


 

## now loop through the above array
for i in "${eras[@]}"
do
   echo ${eras[$x]}

   echo ${list[$x]}
   
   python plot_prediction_deposit_cartopy.py  results_depositsprecip era"$i"results.csv coal   # works  
   python plot_prediction_deposit_cartopy.py  results_depositsprecip era"$i"results.csv evaporites   # works 
   python plot_prediction_deposit_cartopy.py  results_depositsprecip era"$i"results.csv glacial   # works 
 

   python plot_prediction_precitipation_cartopy.py   results_depositsprecip   era"$i"results.csv ${list[$x]}   #    works 
 
   x=$(( $x + 1 ))

done


 

