 
function predict_missingdeposits_(input_file, output_file, output_fileinput,accuracyfile)
 
 
mkdir 14Ma/results
  
 state=sum(100*clock);
 rand('state', state);
 randn('state', state);
 nloop=10000;
 nwarmup=2000;
 %data=readtable('data/miocene_v1.csv');Input file needs to have following
 %strcutre
 %c1=region id; c2=lat; c3=long; c4=coal;c5=evap;c6=glacial;
 %c7=Elev; c8=sqrt(elev); c9=(Dist2Shore); c10=(sqrt(Dist2Shore));
 %c11=sin_h;c12=cos_h xxx
 
 data=readtable(input_file);
 data=table2array(data)
% load('var_names.mat');
 y_index=[4 5 6]% read coal evap glac
 %number_sites_deposit=sum(data(:,y_index),2);
 miss_index=find(data(:,y_index(1))==-999);%missingvalues coded as having deposit of -999
 nmiss=length(miss_index);
 ndep=length(y_index);
 ntot=size(data,1);
 y=data(:,y_index);%cubed root transformation
 obs_index=find(data(:,y_index(1))==-999);
 nobs=length(obs_index);
 y_obs=y(obs_index,:);
 cont_xindex=[7 9];% column of continuous variabels other than lat and lon, currently has evelavtion and dist to shore
 nvar_cont=length(cont_xindex);
 space_xindex=[2,3];%columns of lat and lon
 lin_index=[cont_xindex space_xindex];
 Xspace=data(:,space_xindex);
 nvar=length(lin_index);
 %cov_names=var_names(lin_index);
 Xlin=data(:,lin_index);
 Xlin=[ones(ntot,1),Xlin];
 %standardizing continous predictors
 Xlin(:,2:nvar+1)=(Xlin(:,2:nvar+1)-min(Xlin(:,2:nvar+1)))./range(Xlin(:,2:nvar+1));
  
  
  
 miss_index=find(sum(data(:,y_index),2)==0);
 nmiss=length(miss_index);
 obs_index=find(sum(data(:,y_index),2)>0);
 nobs=length(obs_index);
 for k=1:ndep
  
  
    y=data(:,y_index(k));%
    y_obs=y(obs_index);
    y_miss=y(miss_index);
  
  
  
  
 Xlin_obs=Xlin(obs_index,:);
 Xlin_miss=Xlin(miss_index,:);
 %GP Kernel
 lat=Xlin(:,nvar);
 lon=Xlin(:,nvar+1);
 lat_obs=lat(obs_index)
 lon_obs=lon(obs_index)
 lat_miss=lat(miss_index)
 lon_miss=lon(miss_index)
 Xspace_obs=[lat_obs,lon_obs];
 Xspace_miss=[lat_miss,lon_miss];
   
 %Generating covariance matrices
 [omega_obs]=thinplate_basis(lat_obs,lon_obs,lat_obs,lon_obs);
 [omega_cov_obs_miss]=thinplate_basis(lat_obs,lon_obs,lat_miss,lon_miss);
 [omega_miss]=thinplate_basis(lat_miss,lon_miss,lat_miss,lon_miss);
  cov_invvar_miss_obs=omega_cov_obs_miss'/omega_obs;
    %Representing the Kernal as a linear combination of basis functions
      [Q D]=eig(omega_obs);
      XX_obs=Q*D^.5;
      [Q D]=eig(omega_miss);
      XX_miss=Q*D^.5;
      nbasis=50;
      zmat_obs=[Xlin_obs XX_obs(:,1:nbasis)];
      zmat_miss=[Xlin_miss XX_miss(:,1:nbasis)];
      np=size(zmat_obs,2);
      nfixed=np-nbasis;
        %PUT IN SENSIBLE HYPERPARAMS
      tau_prior_a=-1;
      tau_prior_b=0;
    
%create storage space 
Beta_star_all=zeros(np,nloop+nwarmup);
Sigsq_all=zeros(1,nloop+nwarmup);
Tau_Spline_all=zeros(1,nwarmup+nloop);
Y_fit_all=zeros(nobs,nloop+nwarmup);
log_likelihood_all=zeros(1,nloop+nwarmup);
sigsq_gampar_a=zeros(nloop+nwarmup,1);
sigsq_gampar_b=zeros(nloop+nwarmup,1);
fit_obs_all=zeros(nobs,nloop);
fit_miss_all=zeros(nmiss,nloop);
    %initialise parameters
c1=nobs;
tau_spline=10000;
Tau_Beta=[c1*ones(nfixed,1);tau_spline*ones(nbasis,1)];
Beta_star=zeros(np,1);
zdashz_obs=zmat_obs'*zmat_obs;
 zzinv_obs=inv(zdashz_obs);
% fit_hat_obs=zeros(nobs,ndep);
% fit_hat_miss=zeros(nmiss,ndep);
% low_cred_lim_obs=zeros(nobs,ndep);
% up_cred_lim_obs=zeros(nobs,ndep);
% low_cred_lim_miss=zeros(nmiss,ndep);
% up_cred_lim_miss=zeros(nmiss,ndep);
% classify_nobs=zeros(nobs,ndep);
% classify_miss=zeros(nmiss,ndep);
% classify_all=zeros(ntot,ndep);
% misclassify_rate=zeros(ndep,1);
% fit_hat=zeros(ntot,ndep);
% low_cred_lim=zeros(ntot,ndep);
%  up_cred_lim=zeros(ntot,ndep);
%  for k=1:ndep
%     Tau_Beta=[c1*ones(nfixed,1);tau_spline*ones(nbasis,1)];
%     Beta_star=zeros(np,1);
   % zero_index=find(y_obs(:,k)==0);
   % one_index=find(y_obs(:,k)==1);
     
    zero_index=find(y_obs==0);
    one_index=find(y_obs==1);
    uplim=zeros(nobs,1);
    uplim(one_index)=1000;
    lowlim=zeros(nobs,1);
    lowlim(zero_index)=-1000;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     GIBBS LOOP
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         tic
 
        for p=1:nwarmup + nloop%For 1
            if(mod(p,100)==0)
                p;
                toc
            end
             %Drawing constrained normals
             w_mean=zmat_obs*Beta_star;
             w_var=ones(nobs,1);
             u=rand(nobs,1);
             Fb=normcdf(uplim,w_mean,ones(nobs,1));
             Fa=normcdf(lowlim,w_mean,ones(nobs,1));
             uu=u.*(Fb-Fa)+Fa;
             w=norminv(uu,w_mean,ones(nobs,1));
            %Drawing the expected value fit
           prior_beta_star_mean=zeros(np,1);
           prior_beta_star_prec=diag(1./Tau_Beta);
           like_beta_star_prec=zdashz_obs;
           like_beta_star_mean=zzinv_obs*zmat_obs'*w;
           post_beta_star_prec=prior_beta_star_prec+like_beta_star_prec;
           post_beta_star_var=post_beta_star_prec\eye(np);
           post_beta_star_mean=post_beta_star_var*(like_beta_star_prec*like_beta_star_mean+prior_beta_star_prec*prior_beta_star_mean);
           %post_beta_star_var=0.5*(post_beta_star_var+post_beta_star_var');
           post_beta_star_var=chol(post_beta_star_var)'*chol(post_beta_star_var);
         
           Beta_star=mvnrnd(post_beta_star_mean,post_beta_star_var)';
           fit_obs=zmat_obs*Beta_star;%smooth fit at observed x
           wdev=w-fit_obs;
           fit_miss=Xlin_miss*Beta_star(1:nvar+1)+cov_invvar_miss_obs...
         *(fit_obs-Xlin_obs*Beta_star(1:nvar+1)); 
           %Drawing tausq 
           tau_beta_as=(np-nfixed-1)/2+tau_prior_a;
           tau_beta_bs=sum(Beta_star(nfixed+1:np).^2)/2+tau_prior_b;
           tau_spline=1/gamrnd(tau_beta_as,1./tau_beta_bs);
           Tau_Beta=[c1*ones(nfixed,1);tau_spline*ones(nbasis,1)]; 
           Beta_star_all(:,p)=Beta_star;
           Tau_Spline_all(p)=tau_spline;
           fit_obs_all(:,p)=normcdf(fit_obs);
           fit_miss_all(:,p)=normcdf(fit_miss);
        end 
%     
%     fit_hat_obs=mean(fit_obs_all(:,nwarmup:nloop),2);
%     fit_hat_miss=mean(fit_miss_all(:,nwarmup:nloop),2);
%     fit_hat_miss=mean(fit_miss_all(:,nwarmup:nloop),2);
%     low_cred_lim_obs=quantile(fit_obs_all(:,nwarmup:nloop),0.05,2);
%     up_cred_lim_obs=quantile(fit_obs_all(:,nwarmup:nloop),0.95,2);
%     low_cred_lim_miss=quantile(fit_miss_all(:,nwarmup:nloop),0.05,2);
%     up_cred_lim_miss=quantile(fit_miss_all(:,nwarmup:nloop),0.95,2);
 
 
    fit_hat_obs(:,k)=mean(fit_obs_all(:,nwarmup:nloop),2);
    fit_hat_miss(:,k)=mean(fit_miss_all(:,nwarmup:nloop),2);
    low_cred_lim_obs(:,k)=quantile(fit_obs_all(:,nwarmup:nloop),0.05,2);
    up_cred_lim_obs(:,k)=quantile(fit_obs_all(:,nwarmup:nloop),0.95,2);
    low_cred_lim_miss(:,k)=quantile(fit_miss_all(:,nwarmup:nloop),0.05,2);
    up_cred_lim_miss(:,k)=quantile(fit_miss_all(:,nwarmup:nloop),0.95,2);
     
     
%     
%     classify_nobs=zeros(nobs,1);
%     classify_nobs(fit_hat_obs>0.5)=1;
%     classify_miss=zeros(nmiss,1);
%     classify_miss(fit_hat_miss>0.5)=1;
%     classify_all(obs_index)=classify_nobs;
%     classify_all(miss_index)=classify_miss;
 
 
    classify_nobs(fit_hat_obs(:,k)>0.5,k)=1;
    classify_miss(fit_hat_miss(:,k)>0.5,k)=1;
    classify_all(obs_index,k)=classify_nobs(:,k);
    classify_all(miss_index,k)=classify_miss(:,k);
     
     
%     
%     misclassify_rate(k)=length(find((y_obs-classify_nobs)~=0))/nobs;
%     fit_hat=zeros(ntot,1);
%     fit_hat(obs_index)=fit_hat_obs;
%     fit_hat(miss_index)=fit_hat_miss;
%     low_cred_lim=zeros(ntot,1);
%     up_cred_lim=zeros(ntot,1);
%     low_cred_lim(obs_index)=low_cred_lim_obs;
%     low_cred_lim(miss_index)=low_cred_lim_miss;
%     up_cred_lim(obs_index)=up_cred_lim_obs;
%     up_cred_lim(miss_index)=up_cred_lim_miss;
 
    %misclassify_rate(k)=length(find((y_obs(:,k)-classify_nobs(:,k))~=0))/nobs;
    misclassify_rate(k)=length(find((y_obs-classify_nobs)~=0))/nobs;
    fit_hat(obs_index,k)=fit_hat_obs(:,k);
    fit_hat(miss_index,k)=fit_hat_miss(:,k);
    low_cred_lim(obs_index,k)=low_cred_lim_obs(:,k);
    low_cred_lim(miss_index,k)=low_cred_lim_miss(:,k);
    up_cred_lim(obs_index,k)=up_cred_lim_obs(:,k);
    up_cred_lim(miss_index,k)=up_cred_lim_miss(:,k);
      
      
    if k==1
        %csvwrite('coal_results',results)
        coal_input=y;% for predicting precip
        coal_input(miss_index)=fit_hat(miss_index,k);%using predictedvalues for missing obs
        %csvwrite('coal_input',coal_input)% writing to file
    elseif k==2
        %csvwrite('evaporite_results',results)
        evaporite_input=y;
        evaporite_input(miss_index)=fit_hat(miss_index,k);
        %csvwrite('evaporite_input',evaporite_input)
    else
        %csvwrite('glacial_results',results)
        glacial_input=y;
        glacial_input(miss_index)=fit_hat(miss_index,k);
        %csvwrite('glacial_input',glacial_input)
    end
     
     
 end 
  
  
 csvwrite(accuracyfile,misclassify_rate);
  
  
 deposit_input =[Xspace coal_input evaporite_input glacial_input Xlin(:,[2 3])];
  
 results=[Xspace data(:,y_index)  coal_input evaporite_input glacial_input  fit_hat low_cred_lim up_cred_lim  Xlin(:,[2 3]) ];%writing results to file, first lat lon then deposit recording, the probability of a deposit(fit_hat), with lower and upper cred intervals
   
  
 
 
%results=[deposit_input fit_hat low_cred_lim up_cred_lim];
results(isnan(results))=-999; 
 
 csvwrite(output_file,results);
  
 csvwrite(output_fileinput,deposit_input);
 
 
 end