 clear
 state=sum(100*clock);
 rand('state', state);
 randn('state', state);
 nloop=5000;
 nwarmup=2000;
 %Training Data from Miocene
 load('Xlin_14Ma_input')
 precip_train=readtable('miocene_precip_v1.csv');
 y_index=3;%log prec
 y_obs=table2array(precip_train(:,y_index));
 y_obs=y_obs.^0.33;
 nobs=length(y_obs);
 train_dep_index=[2 3 4];
 ndep=length(train_dep_index);
 train_cont_index=[5 6 7 8];
 ncont=length(train_cont_index);
 train_space_index=[9 10];
 train_index=[train_dep_index train_cont_index train_space_index];
 load('varnames')
 varnames=varnames([train_dep_index-1 train_cont_index-1]);
 varnames=char(varnames);
 Xlin_obs=[ones(nobs,1) Xlin_14Ma_input(:,train_index)];
  
 nvar=size(Xlin_obs,2)-1;
 Xlin_obs(:,ndep+2:nvar+1)=(Xlin_obs(:,ndep+2:nvar+1)-min(Xlin_obs(:,ndep+2:nvar+1)))...
     ./range(Xlin_obs(:,ndep+2:nvar+1));%scaling
 Xlin_obs(:,ndep+2:nvar+1)=Xlin_obs(:,ndep+2:nvar+1).^0.5;
 
 
 %Test Data
 
 test_data=readtable('learning_data_eocene_deposit_2PIC.csv');
 precip_test=readtable('precip_only.csv');
 test_predictor_data=table2array(test_data);
 test_space_index=[2,3];
 test_cont_index=[7 9 11 12];
 test_dep_index=[4 5 6];
 test_index=[test_dep_index test_cont_index test_space_index];
 precip_test=table2array(precip_test);
 y_index=3;%log precip
 y_miss=precip_test(:,y_index).^.33;
 nmiss=length(y_miss);
 ntot=nmiss+nobs;
 Xlin_miss=[ones(nmiss,1) test_predictor_data(:,test_index)]; 
 Xlin_miss(:,ndep+2:nvar+1)=(Xlin_miss(:,ndep+2:nvar+1)-min(Xlin_miss(:,ndep+2:nvar+1)))...
     ./range(Xlin_miss(:,ndep+2:nvar+1));%scaling
 %selecting test(miss) abd train(obs) data
 Xlin_miss(:,ndep+2:nvar+1)=Xlin_miss(:,ndep+2:nvar+1).^0.5
 obs_index=1:nobs;
 miss_index=1:nmiss;
 miss_index=miss_index+nobs;
 y=[y_obs;y_miss];
 Xlin=[Xlin_obs;Xlin_miss];
 lat=Xlin(:,nvar);
 lon=Xlin(:,nvar+1);
 lat_obs=lat(obs_index);
 lon_obs=lon(obs_index);
 lat_miss=lat(miss_index);
 lon_miss=lon(miss_index);
 
figure
for j=1:ncont
    subplot(2,2,j)
    xdata=[ones(ntot,1),Xlin(:,j+ndep+1)];
    ydata=y;
    fit=xdata*(inv(xdata'*xdata))*xdata'*ydata;
    plot(xdata(:,2),ydata,'*')
    title(['Cube Root Precipitation vs ',varnames(j+ndep,:)])
    %lsline
    line(xdata(:,2),fit,'LineWidth',2,'Color',[1 0 0]);
end

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
nbasis=60;
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

results=[test_predictor_data(:,test_space_index) miss_fit_hat low_cred_lim_miss up_cred_lim_miss]
csvwrite('prec_results_38mA',results)
figure
subplot(2,2,1)
xmin=min(fit_all(obs_index)');
xmax=max(fit_all(obs_index)');
rmse_train=sqrt(sum((fit_all(obs_index)'-y(obs_index)).^2)/nobs)
rmse_test=sqrt(sum((fit_all(miss_index)'-y(miss_index)).^2)/nmiss)
plot(fit_all(obs_index)',fit_all(obs_index)'-y(obs_index),'.b')
xlim([xmin xmax])
line(fit_all(obs_index)',zeros(nobs,1),'LineWidth',1,'Color',[0 1 0])
hold
plot(fit_all(miss_index)',fit_all(miss_index)'-y(miss_index),'.r')
title(['Residuals vs Predicted'])
subplot(2,2,2)
plot(fit_all(obs_index),y(obs_index),'.b');
xmin=min(fit_all(obs_index)');
xmax=max(fit_all(obs_index)');
ymin=min(y(obs_index)');
ymax=max(y(obs_index)');
xlim([xmin xmax]);
ylim([ymin ymax])
hold
line(fit_all(obs_index),fit_all(obs_index),'LineWidth',2,'Color',[0 1 0])
plot(fit_all(miss_index),y(miss_index),'.r');
xmin=min(fit_all(miss_index)');
xmax=max(fit_all(miss_index)');
ymin=min(y(miss_index)');
ymax=max(y(miss_index)');
xlim([xmin xmax]);
ylim([ymin ymax])
title(['Predicted vs Actual'])
line(fit_all(miss_index),fit_all(miss_index),'LineWidth',2,'Color',[0 1 0])
subplot(2,2,3)
histfit(y-fit_all',40,'tlocationscale')
title(['Histogram of Residuals with t_3 distribution in red'])
subplot(2,2,4)
boxplot(Beta_star_all(2:nvar-1,nwarmup:nloop)',varnames)%,'varnames(1:nvar-3)');
annotation('line',[0 1],[0 0])
title(['Boxplot of Regression Coefficients'])

% optional, could help make the plot look nicer
% lat_data=(X_space(:,1)-min(X_space(:,1)))./(max(X_space(:,1))-min(X_space(:,1)));
% lon_data=(X_space(:,2)-min(X_space(:,2)))./(max(X_space(:,2))-min(X_space(:,2)));
% % figure
% 
%  tempx_grid1=linspace(min(lat_data),max(lat_data),ngrid)';
%  tempy_grid1=linspace(min(lon_data),max(lon_data),ngrid)';
%  fit_hat1=reshape(fit_hat,ngrid,ngrid);
%  figure
%  surf(tempy_grid1,tempx_grid1,fit_hat1)
%  title(['Fit on Grid assuming all covariates=0 '])
%  hold
%  plot3(lat_data,lon_data,mean_fit,'*')

