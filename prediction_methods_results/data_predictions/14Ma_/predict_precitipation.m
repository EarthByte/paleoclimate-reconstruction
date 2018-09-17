 clear
 state=sum(100*clock);
 rand('state', state);
 randn('state', state);
 nloop=5000;
 nwarmup=2000;
 data=readtable('data/miocene_v1.csv');
 deposit_data=readtable('deposit_input14Ma_v1.csv');
 precip=readtable('data/miocene_precip_v1.csv');
 predictor_data=table2array(data);
 deposit_data=table2array(deposit_data);
 pred_deposit_index=[4 5 6];
 deposit_index=[3 4 5];
 ndep=length(deposit_index);
 predictor_data(:,pred_deposit_index)=deposit_data(:,deposit_index);
 space_index=[2,3];
 cont_index=[7 9];
 precip=table2array(precip);
 y_index=4;%log precip
 y=precip(:,y_index);
 ntot=length(y);
 X_deposit=predictor_data(:,pred_deposit_index);
 X_space=predictor_data(:,space_index);
 X_cont=predictor_data(:,cont_index);
 Xlin_index=[pred_deposit_index cont_index space_index];
 Xlin=[ones(ntot,1) predictor_data(:,Xlin_index)];
 varnames=data.Properties.VariableNames(Xlin_index);
 save varnames varnames
 nvar=size(Xlin,2)-1;
 Xlin(:,ndep+2:nvar+1)=(Xlin(:,ndep+2:nvar+1)-min(Xlin(:,ndep+2:nvar+1)))...
     ./range(Xlin(:,ndep+2:nvar+1));%scaling
 %selecting test(miss) abd train(obs) data
 Xlin_14Ma_input=Xlin%defining precdictor data for other time periods
 csvwrite('Xlin_14Ma_input_v1.csv',Xlin_14Ma_input);
 save  Xlin_14Ma_input  Xlin_14Ma_input
 index_tot=1:ntot;
 index_tot=index_tot';
 nmiss=round(ntot*0.04);
 miss_index=sort(randperm(ntot,nmiss))';
 nobs=ntot-nmiss;
 obs_index=setdiff(index_tot,miss_index);
 y_obs=y(obs_index);
 y_miss=y(miss_index);
 Xlin_obs=Xlin(obs_index,:);
 Xlin_miss=Xlin(miss_index,:);
 lat=Xlin(:,nvar);
 lon=Xlin(:,nvar+1);
 lat_obs=lat(obs_index);
 lon_obs=lon(obs_index);
 lat_miss=lat(miss_index);
 lon_miss=lon(miss_index);
 

 %GP kernel across training

 %[omega_all]=thinplate_basis(lat_all,lon_all,lat_all,lon_all);
 [omega_miss]=thinplate_basis(lat_miss,lon_miss,lat_miss,lon_miss);
 [omega_obs]=thinplate_basis(lat_obs,lon_obs,lat_obs,lon_obs);
 [omega_obs_miss]=thinplate_basis(lat_obs,lon_obs,lat_miss,lon_miss);
 omega_cov_miss_obs=omega_obs_miss'/omega_obs;
%omega_cov_grid_design=omega_data_grid'/omega;
%Representing the Kernal as a linear combination of basis functions
[Q D]=eig(omega_obs);
XX_obs=Q*D^.5;
[Q D]=eig(omega_miss);
XX_miss=Q*D^.5;
nbasis=40;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     GIBBS LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

results=[X_space y fit_all' low_cred_lim_all' up_cred_lim_all']
csvwrite('prec_results_14mA',results)
figure
subplot(2,2,1)
hist(sqrt(Scale_all(nwarmup:nloop)),50)
title(['Distribition of RMSE'])
rmse_train=sum((fit_all(obs_index)'-y(obs_index)).^2)
rmse_test=sum((fit_all(miss_index)'-y(miss_index)).^2)
subplot(2,2,2)
plot(fit_all(obs_index),y(obs_index),'*r');
hold
plot(fit_all(miss_index),y(miss_index),'*b');
title(['Predicted vs Actual'])
lsline
subplot(2,2,3)
histfit(y-fit_all',40,'tlocationscale')
title(['Histogram of Residuals with t_3 distribution in red'])
subplot(2,2,4)
boxplot(Beta_star_all(2:nvar-2,nwarmup:nloop)')%'labels','varnames(1:nvar-3)');
title(['Boxplot of Regression Coefficients'])
% optional, could help make the plot look nicer
lat_data=(X_space(:,1)-min(X_space(:,1)))./(max(X_space(:,1))-min(X_space(:,1)));
lon_data=(X_space(:,2)-min(X_space(:,2)))./(max(X_space(:,2))-min(X_space(:,2)));
% figure
% 
%  tempx_grid1=linspace(min(lat_data),max(lat_data),ngrid)';
%  tempy_grid1=linspace(min(lon_data),max(lon_data),ngrid)';
%  fit_hat1=reshape(fit_hat,ngrid,ngrid);
%  figure
%  surf(tempy_grid1,tempx_grid1,fit_hat1)
%  title(['Fit on Grid assuming all covariates=0 '])
%  hold
%  plot3(lat_data,lon_data,mean_fit,'*')

