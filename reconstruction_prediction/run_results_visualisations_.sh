
#!/bin/sh 
echo Running all 	


## declare an array variable 

#  "38Ma_2PIC" "38Ma_4PIC" 

#declare -a arr=("14Ma_" "28Ma_" "38Ma2PIC_" "38Ma4PIC_" "51Ma_" "61Ma_" "77Ma_" "101Ma_"  "129Ma_"  "154Ma_"  "182Ma_" "219Ma_" "242Ma_" "249Ma_" )

declare -a list=("miocene"  "eocene" "eocene" "other" "other" "other" "other" "other" "other" "other" "other" "other" "other" "other" )


python split_eras.py 


x=0


declare -a eras=("14" "38" "39" "40" "51" "61" "77" "101" "129" "154" "182" "219" "242")



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


 

