

 
function predict_precitipation_others( output_fileinput, infile, infile_testdeposit, output_file )
 

 state=sum(100*clock);
 rand('state', state);
 randn('state', state);
 nloop=500;
 nwarmup=200; 
 
  
 deposit_data=readtable(output_fileinput); 
 precip=readtable(infile); 
 
 
 deposit_data=table2array(deposit_data);
 
 predictor_data=deposit_data;
  
 
 pred_deposit_index=[3 4 5];% index for deposit for predictor_data matrix 
 ndep= 3;
 %predictor_data(:,pred_deposit_index)=deposit_data(:,deposit_index);%putting depot predictions for missing values
 space_index=[1 2 ];%index of lat lon in predictor_data
 cont_index=[6 7];%index of elevation and dist to shore in predictor_data
 precip=table2array(precip);
 y_index=4;%precip
 y_obs=precip(:,y_index).^.33;%transforming using cube root
 nobs=length(y_obs);
 %X_deposit=predictor_data(:,pred_deposit_index);
 X_space=predictor_data(:,space_index);
 %X_cont=predictor_data(:,cont_index);
 Xlin_index=[pred_deposit_index cont_index  ];
 Xlin_obs=[ones(nobs,1) predictor_data(:,Xlin_index)]; 
 
 nvar=size(Xlin_obs,2)-1
 
 Xlin_obs(1:5,:)
 
 
 Xlin_obs(:,ndep+2:nvar+1)=(Xlin_obs(:,ndep+2:nvar+1)-min(Xlin_obs(:,ndep+2:nvar+1)))...
    ./range(Xlin_obs(:,ndep+2:nvar+1));%scaling


 Xlin_obs(1:5,:)
  
 Xlin_obs(:,ndep+2:nvar+1)=Xlin_obs(:,ndep+2:nvar+1).^0.5;
 
 
 Xlin_obs(1:5,:)
 
% Test Data
 
 test_data=readtable(infile_testdeposit);
 test_predictor_data=table2array(test_data);
 test_space_index=[1 2];
 
 X_space_test=test_predictor_data(:,test_space_index);
 
 test_cont_index=[6 7];
 test_dep_index=[3 4 5 ];
 test_index=[test_dep_index test_cont_index  ];
 
 nmiss=size(test_data,1);
  
  
  
 
  ntot=nmiss+nobs;
  
  Xlin_miss=[ones(nmiss,1) test_predictor_data(:,test_index)];
%   
 
 Xlin_miss(:,ndep+2:nvar+1)=(Xlin_miss(:,ndep+2:nvar+1)-min(Xlin_miss(:,ndep+2:nvar+1)))...
    ./range(Xlin_miss(:,ndep+2:nvar+1));%scaling

   

   %selecting test(miss) abd train(obs) data
 Xlin_miss(:,ndep+2:nvar+1)=Xlin_miss(:,ndep+2:nvar+1).^0.5;
 obs_index=1:nobs;
 miss_index=1:nmiss;
 miss_index=miss_index+nobs;
 Xlin=[Xlin_obs;Xlin_miss];
 lat=Xlin(:,nvar);
 lon=Xlin(:,nvar+1);
 lat_obs=lat(obs_index);
 lon_obs=lon(obs_index);
 lat_miss=lat(miss_index);
 lon_miss=lon(miss_index);
 
 
 Xlin_miss(1:5,:)

  %%%%%
 
 [omega_miss]=thinplate_basis(lat_miss,lon_miss,lat_miss,lon_miss);
  
 [omega_obs]=thinplate_basis(lat_obs,lon_obs,lat_obs,lon_obs);
 %omega_obs
 [omega_obs_miss]=thinplate_basis(lat_obs,lon_obs,lat_miss,lon_miss);
 omega_cov_miss_obs=omega_obs_miss';%/omega_obs % THIS GIVES NAN ERROR!
  
 
%Representing the Kernal as a linear combination of basis functions
[Q D]=eig(omega_obs);
XX_obs=Q*D^.5;
[Q D]=eig(omega_miss);
XX_miss=Q*D^.5;
nbasis=100;%**********play with this
zmat_obs=[Xlin_obs XX_obs(:,1:nbasis)];
zmat_miss=[Xlin_miss XX_miss(:,1:nbasis)];
np=size(zmat_obs,2);
nfixed=np-nbasis;
z_miss=Xlin_miss;
%PUT IN SENSIBLE HYPERPARAMS
tau_prior_a=-1;
tau_prior_b=0;
sigsq_prior_a=1;
sigsq_prior_b=1;
%create storage space 
Y_obs_fit_all=zeros(nobs,nloop+nwarmup);
Y_miss_fit_all=zeros(nmiss,nloop+nwarmup);
sigsq_gampar_a=zeros(nloop+nwarmup,1);
sigsq_gampar_b=zeros(nloop+nwarmup,1);
Beta_star_all=zeros(np,nloop+nwarmup);
Scale_all=zeros(1,nloop+nwarmup);
Tau_Spline_all=zeros(1,nwarmup+nloop);
 %initialise parameters
c1=ntot/4;
tau_spline=10000;
scale=var(y_obs)/4;
kappa=ones(nobs,1);
Sigsq_inv=diag(kappa./scale);
Sigsq=diag(scale./kappa);
Tau_Beta=[c1*scale*ones(nfixed,1);tau_spline*ones(nbasis,1)];
Beta_star=zeros(np,1);
zdashz_obs=zmat_obs'*zmat_obs;
zzinv_obs=inv(zdashz_obs);
nu=3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    GIBBS LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
for p=1:nwarmup + nloop%For 1
    if(mod(p,100)==0)
        p;
        toc
    end
                        
    %Drawing the expected value fit
    prior_beta_star_mean=zeros(np,1);
    prior_beta_star_prec=diag(1./Tau_Beta);
    like_beta_star_prec=zmat_obs'*Sigsq_inv*zmat_obs;
    like_beta_star_mean=(zmat_obs'*Sigsq_inv*zmat_obs)\zmat_obs'*Sigsq_inv*y_obs;
    post_beta_star_prec=prior_beta_star_prec+like_beta_star_prec;
    post_beta_star_var=post_beta_star_prec\eye(np);
    post_beta_star_mean=post_beta_star_var*(like_beta_star_prec*like_beta_star_mean+prior_beta_star_prec*prior_beta_star_mean);
    post_beta_star_var=chol(post_beta_star_var)'*chol(post_beta_star_var);
    Beta_star=mvnrnd(post_beta_star_mean,post_beta_star_var)';
    yfit_obs=zmat_obs*Beta_star;%smooth fit at observed x
    ydev_obs=y_obs-yfit_obs;
    %Predicting out of sample
    Beta_lin=Beta_star(1:nvar+1);
    yfit_miss=Xlin_miss*Beta_lin+omega_cov_miss_obs...
         *(yfit_obs-Xlin_obs*Beta_star(1:nvar+1));
     % Drawing Sigmasq 
     %Drawing "weights"kappa^-1
     kappa_gampar_b=(nu*ones(nobs,1)+ydev_obs.^2/scale)./2;
     kappa_gampar_a=ones(nobs,1).*(1+nu)/2;
     kappa=gamrnd(kappa_gampar_a,1./kappa_gampar_b);
     %Drawing scale
     sigsq_gampar_a=(nobs+sigsq_prior_a)/2;
     sigsq_gampar_b=(sigsq_prior_b+sum(ydev_obs.^2./kappa))/2;
     scale=1./gamrnd(sigsq_gampar_a,1./sigsq_gampar_b);
     Sigsq_inv=diag(kappa./scale);
     Sigsq=diag(scale./kappa);  
     %Drawing Tau
     tau_beta_as=(np-nfixed-1)/2+tau_prior_a;
     tau_beta_bs=sum(Beta_star(nfixed+1:np).^2)/2+tau_prior_b;
     tau_spline=1/gamrnd(tau_beta_as,1./tau_beta_bs);
     Tau_Beta=[c1*scale*ones(nfixed,1);tau_spline*ones(nbasis,1)]; 
        %Storing values   
     Beta_star_all(:,p)=Beta_star;
     Scale_all(p)=scale;
     Tau_Spline_all(p)=tau_spline;
     Y_obs_fit_all(:,p)=yfit_obs; 
     Y_miss_fit_all(:,p)=yfit_miss;
end
toc
%Graphs
obs_fit_hat=mean(Y_obs_fit_all(:,nwarmup:nloop),2);
miss_fit_hat=mean(Y_miss_fit_all(:,nwarmup:nloop),2);
fit_all(obs_index)=obs_fit_hat';
fit_all(miss_index)=miss_fit_hat';
low_cred_lim_obs=quantile(Y_obs_fit_all(:,nwarmup:nloop),0.05,2);
up_cred_lim_obs=quantile(Y_obs_fit_all(:,nwarmup:nloop),0.95,2);
low_cred_lim_miss=quantile(Y_miss_fit_all(:,nwarmup:nloop),0.05,2);
up_cred_lim_miss=quantile(Y_miss_fit_all(:,nwarmup:nloop),0.95,2);
fit_all(obs_index)=obs_fit_hat';
fit_all(miss_index)=miss_fit_hat';
low_cred_lim_all(obs_index)=low_cred_lim_obs';
low_cred_lim_all(miss_index)=low_cred_lim_miss';
up_cred_lim_all(obs_index)=up_cred_lim_obs';
up_cred_lim_all(miss_index)=up_cred_lim_miss';
  
results=[X_space_test miss_fit_hat low_cred_lim_miss up_cred_lim_miss]
 csvwrite(output_file,results) 
 
 
  

end

