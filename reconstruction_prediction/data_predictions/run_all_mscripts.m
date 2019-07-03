clear
 

% Predict for the rest

x = [ "14", "28", "51", "61", "77", "101", "129", "154", "182", "219", "242", "249"]


for i=1:12
fname =  x(i)+"Ma_/data/predictor_data_land.csv"
outfname =  x(i)+"Ma_/results/deposit_results.csv"
outfname2 =  x(i)+"Ma_/results/deposit_input.csv"
accuracyfile = x(i)+"Ma_/results/mis_class.csv"

%predict_missingdeposits_(fname, outfname, outfname2, accuracyfile)
 

end
 


%miocene_deposits = "14Ma_/data/predictor_data_land.csv"
miocene_precip = "14Ma_/data/miocene_precip.csv" 
predict_file = "14Ma_/results/prec_results.csv" 
outfname2 =  "14Ma_/results/deposit_input.csv"
miocene_knowledge = "Xlin_14Ma_input"

%predict_precitipation_miocene_( outfname2, miocene_precip, predict_file)

 
x_ = [ "51", "61", "77", "101", "129", "154", "182", "219", "242", "249"]
  
for i=1:9
    
  
%infile_deposit = x_(i)+ "Ma_/data/predictor_data_land.csv"

infile_deposit =  x_(i)+ "Ma_/results/deposit_input.csv"
predict_file = x_(i)+ "Ma_/results/prec_results.csv"

predict_precitipation_others( miocene_knowledge, miocene_precip, infile_deposit, predict_file )


%predict_precitipation_githubversion( miocene_knowledge, miocene_precip, infile_deposit, predict_file )

 

end
 
