
#!/bin/sh 
echo Running all 	


## declare an array variable 

#  "38Ma_2PIC" "38Ma_4PIC" 

#declare -a arr=("14Ma_" "28Ma_" "51Ma_" "77Ma_" "101Ma_"  "129Ma_"  "154Ma_"  "182Ma_" "219Ma_" "242Ma_" "249Ma_" )


declare -a arr=("28Ma_")

## now loop through the above array
for i in "${arr[@]}"
do
   echo data_predictions/"$i"
   # or do whatever with individual element of the array
   python plot_prediction_deposit.py data_predictions/"$i"/results  deposit_results.csv coal   # works 
   python plot_prediction_deposit.py data_predictions/"$i"/results  deposit_results.csv evaporites # works 
   python plot_prediction_deposit.py data_predictions/"$i"/results  deposit_results.csv glacial  # works  

   python plot_prediction_precitipation.py  data_predictions/"$i"/results   prec_results.csv precitipation #    works 

done


 

