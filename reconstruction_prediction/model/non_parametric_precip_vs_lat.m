     clear
     state=sum(100*clock);
     rand('state', state);
     randn('state', state);
     nloop=5000;
     nwarmup=5000;
     %Input Miocdene data
     deposit_info=readtable('deposit_input_all_eras_new.csv')
     deposit_info=sortrows(deposit_info,[7,1,2]);
     eocene_index=find(deposit_info.Era==38)
     eocene_deposit=deposit_info(eocene_index,:);
     miocene_index=find(deposit_info.Era==14)
     miocene_deposit=deposit_info(miocene_index,:);
     precip_obs=readtable('precipinput_14Ma_v2_38.csv');% Table of 3 columns lat lon and precip
     precip_obs=table2array(precip_obs);
     precip_obs=sortrows(precip_obs,[4,1,2]);
     miss_index=find(precip_obs(:,4)>38);
     obs_index=find(precip_obs(:,4)==38);
     y_index=3;%column location of preciptation. No missing precipitation results for the training data set  
     y_precip_obs=precip_obs(obs_index,y_index).^0.3;% taking the cube root
     %ntrain=size(precip_obs,1);% using the first two eras as training
     ntrain=length(obs_index);
     %ntest=n-ntrain;
     lat_precip_obs=precip_obs(obs_index,1);
     lat_scaled_obs=(lat_precip_obs-min(lat_precip_obs)+1/100)./range(lat_precip_obs);
     [lat_unique_obs,ia_obs,ic_obs]=unique(lat_scaled_obs);   
     nlat_unique=length(lat_unique_obs);
     y_precip_unique=zeros(nlat_unique,1);
%     
     [omega_lat_precip_obs]=cubic_basis(lat_unique_obs,lat_unique_obs);
     nbasis_lat=10;
     %defining covariance strcuture at obs points. Note extention exp
     %refers to the orginal datase while extension unique refers to the
     %unique values of X
     [Q, D]=eig(omega_lat_precip_obs);
     %nonparametric component ofdesign matrix at the unique values of latitude
     XX_lat_precip_obs_unique=Q*D^.5;
     XX_lat_precip_obs_unique=XX_lat_precip_obs_unique(:,nlat_unique-nbasis_lat+1:nlat_unique);
     Xmat_precip_obs_lin_unique=[ones(nlat_unique,1) lat_unique_obs];
     Xmat_precip_obs_unique=[Xmat_precip_obs_lin_unique XX_lat_precip_obs_unique];

     %nonparametric component ofdesign matrix at all values of latitude
     XX_lat_precip_obs_exp=XX_lat_precip_obs_unique(ic_obs,:);
     Xmat_precip_obs_lin_exp=[ones(ntrain,1) lat_scaled_obs];
     Xmat_precip_obs_exp=[Xmat_precip_obs_lin_exp XX_lat_precip_obs_exp];
     xdashx_exp=Xmat_precip_obs_exp'*Xmat_precip_obs_exp;
     xdashx_inv_exp=inv(xdashx_exp);
     %plot(Xmat_precip_obs_exp*xdashx_inv_exp*Xmat_precip_obs_exp'*y_precip_obs, y_precip_obs,'*')
     ngrid=50;
     nrep=1;
     lat_grid=repmat(linspace(1/ngrid,1,ngrid)',nrep,1);
     [lat_unique_grid,ia_grid,ic_grid]=unique(lat_grid);
     [omega_temp]=cubic_basis(lat_unique_grid,lat_unique_grid);
     omega_precip_lat_grid=omega_temp;
     [cov_grid_obs]=cubic_basis(lat_unique_grid,lat_unique_obs);
     cov_lat_invvar_precip_unique=cov_grid_obs/omega_lat_precip_obs;
     cov_lat_invvar_precip_expand=cov_lat_invvar_precip_unique(ic_grid,:);
     Xmat_precip_miss_lin=[ones(nrep*ngrid,1) lat_grid];
     Xmat_precip_lin_exp=[Xmat_precip_obs_lin_exp;Xmat_precip_miss_lin];
     nvar_precip_lin=size(Xmat_precip_obs_lin_exp,2);%design matrix for predicting precipitation which excludes spatial component
     nvar_precip=nvar_precip_lin+nbasis_lat;
   
     c_beta=ntrain;
     Sigsq=ones(nloop+nwarmup,1);
     Sigsq(1)=var(y_precip_obs)
     Sigsq_inv=diag(1/Sigsq(1));
     Tau_Beta=10000*ones(nloop+nwarmup,1);
     sigsq_prior_a=1;
     sigsq_prior_b=1;
     tau_prior_a=1;
     tau_prior_b=1;
      y_precip_fit_obs=zeros(nlat_unique,nloop+nwarmup+1);
     y_precip_fit_obs_exp=zeros(ntrain,nloop+nwarmup+1);
     y_precip_fit_miss=zeros(ngrid*nrep,nloop+nwarmup+1);
     Beta_star=zeros(nvar_precip,nloop+nwarmup+1);
     beta_hat=xdashx_inv_exp*Xmat_precip_obs_exp'*y_precip_obs;
     Beta_star(:,1)=beta_hat;
     plot(lat_scaled_obs,Xmat_precip_obs_exp*Beta_star(:,1))
     for p=1:nloop+nwarmup
         prior_beta_star_mean=zeros(nvar_precip,1);
         prior_beta_star_prec=diag([ones(nvar_precip_lin,1)./c_beta;ones(nbasis_lat,1)./Tau_Beta(p)]);
         like_beta_star_prec=Sigsq_inv*xdashx_exp;
         like_beta_star_mean=xdashx_inv_exp'*Xmat_precip_obs_exp'*y_precip_obs;
         post_beta_star_prec=prior_beta_star_prec+like_beta_star_prec;
         post_beta_star_var=post_beta_star_prec\eye(nvar_precip);
         post_beta_star_mean=post_beta_star_var*(like_beta_star_prec*like_beta_star_mean+prior_beta_star_prec*prior_beta_star_mean);
         post_beta_star_var=chol(post_beta_star_var)'*chol(post_beta_star_var);
         Beta_star(:,p+1)=mvnrnd(post_beta_star_mean,post_beta_star_var)';
         y_precip_fit_obs_exp(:,p+1)=Xmat_precip_obs_exp*Beta_star(:,p+1);
         ydev_obs=y_precip_obs-y_precip_fit_obs_exp(:,p+1);
         y_precip_fit_miss(:,p+1)=Xmat_precip_miss_lin*Beta_star(1:nvar_precip_lin,p+1)+...
         cov_lat_invvar_precip_unique*(XX_lat_precip_obs_unique*Beta_star(nvar_precip_lin+1:end,p+1));
         
         %drawing sigmasq
         sigsq_gampar_a=(ntrain+sigsq_prior_a)/2;
         sigsq_gampar_b=sigsq_prior_b+sum(ydev_obs.^2)/2;
         Sigsq(p+1)=1./gamrnd(sigsq_gampar_a,1./sigsq_gampar_b);
         Sigsq_inv=diag(1/Sigsq(p+1)); 
         %Drawing tausq_beta 
         tau_beta_ab=(nbasis_lat-1)/2+tau_prior_a;
            tau_beta_bb=sum(Beta_star(nvar_precip_lin+1:nvar_precip,p+1).^2)/2+tau_prior_b;
            Tau_Beta(p+1)=1/gamrnd(tau_beta_ab,1./tau_beta_bb);
     end
     precip_fit_hat_obs_exp=mean(y_precip_fit_obs_exp(:,nwarmup:nloop+nwarmup),2);
     precip_lcl_hat_obs_exp=quantile(y_precip_fit_obs_exp(:,nwarmup:nloop+nwarmup),0.05,2);
     precip_ucl_hat_obs_exp=quantile(y_precip_fit_obs_exp(:,nwarmup:nloop+nwarmup),0.90,2);
     precip_fit_hat_miss=mean(y_precip_fit_miss(:,nwarmup:nloop+nwarmup),2);
     precip_lcl_hat_miss=quantile(y_precip_fit_miss(:,nwarmup:nloop+nwarmup),0.05,2);
     precip_ucl_hat_miss=quantile(y_precip_fit_miss(:,nwarmup:nloop+nwarmup),0.90,2);
     figure
     hold
     plot(lat_precip_obs,y_precip_obs.^(10/3),'*')
     %plot(lat_precip_obs,precip_fit_hat_obs_exp.^(10/3),'*')
  
     lat_grid_raw=lat_grid.*range(lat_precip_obs)-1/100+min(lat_precip_obs)
     x=lat_grid_raw';
     y1=precip_lcl_hat_miss.^(10/3)';
     y2=precip_ucl_hat_miss.^(10/3)';
%      plot(x,y1,'r')
%     
%      plot(x,y2,'r')
%      x2=[x, fliplr(x)];
%      inBetween = [y1, fliplr(y2)];
%      h=fill(x2, inBetween, 'g');
%      patch([x fliplr(x)], [y1 fliplr(y2)], 'r')
%      
%      
%      plot(lat_grid_raw,precip_fit_hat_miss.^(10/3),'r')
     %Plotting Uncertainty
    [ph,msg]=jbfill(x,y1,y2,rand(1,3),rand(1,3),0,0.4);
    figure
    
     gscatter(eocene_deposit.lat,y_precip_obs.^(10/3),eocene_deposit.Dep_Type,'brgbl')
     
     
     
     resid=y_precip_obs-precip_fit_hat_obs_exp;
     figure
     plot(precip_fit_hat_obs_exp,y_precip_obs,'*')
     
     