 clear
 state=sum(100*clock);
 rand('state', state);
 randn('state', state);
 nloop=10000;
 nwarmup=2000;
 data=csvread('data/miocene_v2.csv',1,0);
 %The data format should c1 region id, cols 2 and 3 are lat and lon,
 %coloumns 4,5,6 are coal, evap, glac deposit respectively, columns 7 and 8
 %are elevation and sqrt(elavation), columns 9 and 10 are dist2shore and
 %sqrt(dist2shore), columns 11 and 12 are sin_h, cos_h
 y_index=[4,5,6];%selecting the deposit indexes
 ndep=length(y_index);
 ntot=size(data,1);
 cont_xindex=[7 9];% column of continuous variabels other than lat and lon.  This ocrresponds to elevationa and dist to sharre
 %other options are sqrt(elevation) and sqrt(dist2shore), which are columns
 %8 and 10
 nvar_cont=length(cont_xindex);
 space_xindex=[2,3];%columns of lat and lon
 lin_index=[cont_xindex space_xindex];%regression function is Xlin*beta+f(la,lon), where f is the GP and X is designmatrix of linear preictors
 Xspace=data(:,space_xindex);
 nvar=length(lin_index);
 %cov_names=var_names(lin_index);
 Xlin=data(:,lin_index);
 Xlin=[ones(ntot,1),Xlin];%adding intercept
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
     lat=Xlin(:,nvar);%assumes that the space index, i.e. lat,lon are the last two columns of Xlin, as per lin_index=[cont_xindex space_xindex];%
     lon=Xlin(:,nvar+1);
     lat_obs=lat(obs_index);
     lon_obs=lon(obs_index);
     lat_miss=lat(miss_index);
     lon_miss=lon(miss_index);
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
%         figure
%         subplot(2,2,1)
%         hist(Beta_star_all(2,nwarmup:nloop),50)
%         subplot(2,2,2)
%         hist(Beta_star_all(3,nwarmup:nloop),50)
%         subplot(2,2,3)
%         hist(Beta_star_all(4,nwarmup:nloop),50)
%         subplot(2,2,4)
%         hist(Beta_star_all(5,nwarmup:nloop),50)
    fit_hat_obs=mean(fit_obs_all(:,nwarmup:nloop),2);
    fit_hat_miss=mean(fit_miss_all(:,nwarmup:nloop),2);
    fit_hat_miss=mean(fit_miss_all(:,nwarmup:nloop),2);
    low_cred_lim_obs=quantile(fit_obs_all(:,nwarmup:nloop),0.05,2);
    up_cred_lim_obs=quantile(fit_obs_all(:,nwarmup:nloop),0.95,2);
    low_cred_lim_miss=quantile(fit_miss_all(:,nwarmup:nloop),0.05,2);
    up_cred_lim_miss=quantile(fit_miss_all(:,nwarmup:nloop),0.95,2);
    classify_nobs=zeros(nobs,1);
    classify_nobs(fit_hat_obs>0.5)=1;
    classify_miss=zeros(nmiss,1);
    classify_miss(fit_hat_miss>0.5)=1;
    classify_all(obs_index)=classify_nobs;
    classify_all(miss_index)=classify_miss;
    misclassify_rate(k)=length(find((y_obs-classify_nobs)~=0))/nobs;
    fit_hat=zeros(ntot,1);
    fit_hat(obs_index)=fit_hat_obs;
    fit_hat(miss_index)=fit_hat_miss;
    low_cred_lim=zeros(ntot,1);
    up_cred_lim=zeros(ntot,1);
    low_cred_lim(obs_index)=low_cred_lim_obs;
    low_cred_lim(miss_index)=low_cred_lim_miss;
    up_cred_lim(obs_index)=up_cred_lim_obs;
    up_cred_lim(miss_index)=up_cred_lim_miss;
   
% for j=1:nhist
%     subplot(floor(nhist/2)+1,2,j)
%     hist(Beta_star_all(j+1,nwarmup:nloop),50)
%     title(['Histogram of \beta for', var_names{cont_xindex(j)}])
% end
    results=[Xspace y fit_hat low_cred_lim up_cred_lim];%writing results to file, first lat lon then deposit recording, the probability of a deposit(fit_hat), with lower and upper cred intervals
    if k==1
        csvwrite('coal_results',results)
        coal_input=y;% for predicting precip
        coal_input(miss_index)=fit_hat(miss_index);%using predictedvalues for missing obs
        csvwrite('coal_input',coal_input)% writing to file
    elseif k==2
        csvwrite('evaporite_results',results)
        evaporite_input=y;
        evaporite_input(miss_index)=fit_hat(miss_index);
        csvwrite('evaporite_input',evaporite_input)
    else
        csvwrite('glacial_results',results)
        glacial_input=y;
        glacial_input(miss_index)=fit_hat(miss_index);
        csvwrite('glacial_input',glacial_input)
    end
 end
 deposit_input_all_v2=[Xspace coal_input evaporite_input glacial_input]
 save deposit_input_all_v2 deposit_input_all_v2
 csvwrite('deposit_input_all_v2.csv',deposit_input_all_v2)

%         subplot(2,2,2)
%         hist(Beta_star_all(3,nwarmup:nloop),50)
%         subplot(2,2,3)
%         hist(Beta_star_all(4,nwarmup:nloop),50)
%         subplot(2,2,4)
%         hist(Beta_star_all(5,nwarmup:nloop),50)
