
#!/bin/sh 
echo Running all 	


## declare an array variable 

#  "38Ma_2PIC" "38Ma_4PIC" 

#declare -a arr=("14Ma_" "28Ma_" "38Ma2PIC_" "38Ma4PIC_" "51Ma_" "61Ma_" "77Ma_" "101Ma_"  "129Ma_"  "154Ma_"  "182Ma_" "219Ma_" "242Ma_" "249Ma_" )

declare -a list=("miocene"  "eocene2PIC" "eocene4PIC" "other" "other28" "other51" "other61" "other" "other" "other" "other" "other" "other" "other" )


python split_eras.py 


x=0
#declare -a arr=("14Ma_" "28Ma_")

#declare -a eras=("14" "38" "39" "28")

#14, 38, 28, 51, 61, 77, 101, 129, 154, 182, 219, 242


declare -a eras=("14" "38" "39" "28" "51" "61" "77" "101" "129" "154" "182" "219" "242")

## now loop through the above array
for i in "${eras[@]}"
do
   echo ${eras[$x]}

   echo ${list[$x]}
   
   python plot_prediction_deposit_.py data_prediction_nov2019/results_depositsprecip era"$i"results.csv coal   # works  
   python plot_prediction_deposit_.py data_prediction_nov2019/results_depositsprecip era"$i"results.csv evaporites   # works 
   python plot_prediction_deposit_.py data_prediction_nov2019/results_depositsprecip era"$i"results.csv glacial   # works 

   python plot_prediction_precitipation_.py  data_prediction_nov2019/results_depositsprecip   era"$i"results.csv ${list[$x]}   #    works 
 
   x=$(( $x + 1 ))

done


 

