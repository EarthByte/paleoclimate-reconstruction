clear

% % train for miocence - train purpose 
 folder = "14Ma_"
 fname =  folder+"/data/miocene_v1.csv"
 outfname = folder+"/deposit_results14Ma_v1.csv"
 outfname2 = folder+"/deposit_input14Ma_v1.csv"
% 
 predict_missingdeposits_miocene(fname, outfname, outfname2)
%  
% 
 out_predict = folder+"/results/prec_results.csv"
 infile = folder+"/data/miocene_precip_v1.csv"
% 
 predict_precitipation_miocene(folder, fname, outfname2, infile,  out_predict)

 



% Predict for the rest

x = [ "28", "51", "61", "77", "101", "129", "154", "182", "219", "242", "249"]


 
 
  
%for i=1:11
%fname =  x(i)+"Ma_/data/predictor_data_land.csv"
%outfname =  x(i)+"Ma_/results/deposit_results.csv"
%outfname2 =  x(i)+"Ma_/results/deposit_input.csv"
%accuracyfile = x(i)+"Ma_/results/mis_class.csv"

%predict_missingdeposits(fname, outfname, outfname2, accuracyfile)

%fname_in = x(i)+"Ma_/data/Xlin_14Ma_input.csv"
%fname_in2 = x(i)+"Ma_/data/miocene_precip_v1.csv"
%infile_deposit = x(i)+ "Ma_/data/predictor_data_land.csv"
%predict_file = x(i)+ "Ma_/results/prec_results.csv"

%predict_precitipation(fname_in, fname_in2, infile_deposit, predict_file )

 

%end
 
