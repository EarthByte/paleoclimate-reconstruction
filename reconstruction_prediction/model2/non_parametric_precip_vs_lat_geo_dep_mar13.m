     clear
     state=sum(100*clock);
     rand('state', state);
     randn('state', state);
     nloop=5000;
     nwarmup=5000;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %Input deposit and era data
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     deposit_info=readtable('deposit_input_all_eras_new.csv');
     deposit_info=sortrows(deposit_info,[7,1,2]);
     deposit_var_names=deposit_info.Properties.VariableNames;
     deposit_index=[3 4 5];%location of columns of coal, evaporite and glacial missing values are coded -999 .
     num_dep=length(deposit_index);
     deposit_names=deposit_var_names(deposit_index);
     era_id=deposit_info(:,7);
     era_id=table2array(era_id);
     era_unique=unique(era_id,'stable');
     num_era=length(era_unique);
     deposit_data=deposit_info(:,deposit_index);
     deposit_data=table2array(deposit_data);
     n=length(deposit_data);
     dep_miss_ind=zeros(n,1);
     dep_miss_ind(deposit_data(:,1)==-999)=1;
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %Input geo data
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     geo_info=readtable('geo_input_all_eras_new.csv');
     geo_info=sortrows(geo_info,[9,1,2]);
     geo_index=[5 7 8];
     geo_names=geo_info.Properties.VariableNames;
     geo_names_pred=geo_names(geo_index);
     predict_names=[geo_names_pred deposit_names];
     space_index=[1 2 3];
     space_data=table2array(geo_info(:,space_index));
     space_scaled_data=(space_data-min(space_data)+1/100)./range(space_data);
     lat_scaled=space_scaled_data(:,1);
     lon_scaled=space_scaled_data(:,2);
     elev_scaled=space_scaled_data(:,3);
     geo_data=table2array(geo_info(:,geo_index));
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %Input precipitation data
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     precip=readtable('precipinput_14Ma_v2_38.csv');% Table of 3 columns lat lon and precip
     precip=table2array(precip);
     precip=sortrows(precip,[4,1,2]);
     ntrain=length(precip);% using the first two eras as training
     obs_index=1:ntrain;
     miss_index=ntrain+1:n;
     y_index=3;%column location of preciptation. No missing precipitation results for the training data set  
     y_precip_obs=precip(obs_index,y_index).^0.3;% taking the cube root
     ntest=n-ntrain;
     lat_scaled_obs=lat_scaled(obs_index);
     elev_scaled_obs=elev_scaled(obs_index);
     lat_elev_obs=[lat_scaled_obs, elev_scaled_obs];
      
     era_index=cell(num_era,1);
     dep_obs_index=cell(num_era,1);
     dep_miss_index=cell(num_era,1);
     nobs_dep_index=zeros(num_era,1);
     nmiss_dep_index=zeros(num_era,1);
     n_era=zeros(num_era,1);
     deposit_data_indiv=cell(num_era,1);
     dep_miss_indiv=cell(num_era,1);
     for k=1:num_era
         era_index{k}=find(era_id==era_unique(k));
         n_era(k)=length(era_index{k});
         dep_obs_index{k}=find(era_id==era_unique(k) & dep_miss_ind==0);
         dep_miss_index{k}=find(era_id==era_unique(k) & dep_miss_ind==1);
         deposit_data_indiv{k}=deposit_data(era_index{k},:);
         dep_miss_indiv{k}=find(deposit_data_indiv{k}(:,1)==-999);
         nobs_dep_index(k)=length(dep_obs_index{k});
         nmiss_dep_index(k)=length(dep_miss_index{k});    
     end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Defining covariance stucture at obs points.
      %  1. Extension "expand" refers to the orginal dataset, which has multiple
      %     instances of the same lat/eleve etc
      %  2. Extension "unique" refers to the unique values of X
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     [lat_elev_obs_unique,ia_lat_elev_obs,ic_lat_elev_obs]=unique(lat_elev_obs(1:ntrain,:),'stable','rows');
     nlat_elev_obs_unique=length(lat_elev_obs_unique);
     y_precip_unique=zeros(nlat_elev_obs_unique,1);
     [omega_lat_elev_precip_obs]=thinplate_basis(lat_elev_obs_unique(:,1),lat_elev_obs_unique(:,2),lat_elev_obs_unique(:,1),lat_elev_obs_unique(:,2));
     nbasis_lat_elev=20;
     %Defining covariance strcuture at obs points. Note extention exp
     %refers to the orginal datase while extension unique refers to the
     %unique values of X
     [Q, D]=eig(omega_lat_elev_precip_obs);
     [d,ind] = sort(diag(D));
     D=D(ind,ind);
     Q=Q(:,ind);
     %nonparametric component ofdesign matrix at the unique values of latitude
     XX_lat_elev_precip_obs_unique=Q*D^.5;
     XX_lat_elev_precip_obs_unique=XX_lat_elev_precip_obs_unique(:,nlat_elev_obs_unique-nbasis_lat_elev+1:nlat_elev_obs_unique);
     %Xmat_precip_obs_lin_unique=[ones(nlat_obs_unique,1) lat_obs_unique];
     %Xmat_precip_obs_unique=[Xmat_precip_obs_lin_unique XX_lat_precip_obs_unique];

     %nonparametric component of design matrix at all values of latitude and
     %Elevation
     XX_lat_elev_precip_obs_expand=XX_lat_elev_precip_obs_unique(ic_lat_elev_obs,:);
     Xmat_precip_obs_lin_expand=[ones(ntrain,1) geo_data(obs_index,:) deposit_data(obs_index,:) lat_scaled_obs elev_scaled_obs];
     Xmat_precip_obs_expand=[Xmat_precip_obs_lin_expand XX_lat_elev_precip_obs_expand];
     xdashx_expand=Xmat_precip_obs_expand'*Xmat_precip_obs_expand;
     xdashx_inv_expand=inv(xdashx_expand);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %defining covariance strcuture between the function at obs values
      %f_obs(lat_i,elev_i) and the missing f_miss(lat_j,elev_j).
      %extention "expand" refers to the orginal dataset, which has multiple
      %instances of the same lat/eleve etc
      %extension "unique" refers to the unique values of X
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     cov_lat_invvar_precip_expand=cell(num_era,1);
     Xmat_precip_miss_lin_expand=cell(num_era,1);
     Xmat_precip_expand=Xmat_precip_obs_expand;
     Xmat_precip_miss_expand=cell(num_era,1);
     XX_lat_elev_precip_miss_expand=cell(num_era,1);
     XX_lat_elev_precip_miss_unique=cell(num_era,1);
     cov_lat_invvar_precip_unique=cell(num_era,1);
     for k=1:num_era% 
         %% 
         lat_era{k}=space_scaled_data(era_index{k},1);
         elev_era{k}=space_scaled_data(era_index{k},3);
         lat_elev_miss{k}=[lat_era{k} elev_era{k}];
         [lat_elev_miss_unique,ia_lat_elev_miss,ic_lat_elev_miss]=unique(lat_elev_miss{k},'stable','rows');
         ia_lat_elev_miss_ind{k}=ia_lat_elev_miss;
         ic_lat_elev_miss_ind{k}=ic_lat_elev_miss;
         %% 
         nlat_elev_miss_unique=length(lat_elev_miss_unique);
         [omega_precip_lat_elev_miss]=thinplate_basis(lat_elev_miss_unique(:,1),lat_elev_miss_unique(:,2),lat_elev_miss_unique(:,1),lat_elev_miss_unique(:,2));
         [Q, D]=eig(omega_precip_lat_elev_miss);
         [d,ind] = sort(diag(D));
         D=D(ind,ind);
         Q=Q(:,ind);
     %nonparametric component ofdesign matrix at the unique values of
     %latitude for era k
         XX_lat_elev_precip_miss_unique{k}=Q*D^.5;
         XX_lat_elev_precip_miss_expand{k}=XX_lat_elev_precip_obs_unique(ic_lat_elev_miss_ind{k},:);
         [cov_precip_lat_elev_obs_miss_unique]=thinplate_basis(lat_elev_miss_unique(:,1),lat_elev_miss_unique(:,2),lat_elev_obs_unique(:,1),lat_elev_obs_unique(:,2));
         cov_lat_elev_invvar_precip_unique{k}=cov_precip_lat_elev_obs_miss_unique/omega_lat_elev_precip_obs;
         Xmat_precip_miss_lin_expand{k}=[ones(n_era(k),1)  geo_data(era_index{k},:) deposit_data(era_index{k},:) lat_elev_miss{k}];
         if k > 2
             Xmat_precip_miss_expand=[Xmat_precip_miss_lin_expand{k} XX_lat_elev_precip_miss_expand{k}];
            Xmat_precip_expand=[Xmat_precip_expand; Xmat_precip_miss_expand];
         end
     end
     
     nvar_precip_lin=size(Xmat_precip_obs_lin_expand,2);
     nvar_precip=nvar_precip_lin+nbasis_lat_elev;
     
     %####################################################################
      %Setting up GP Prior and forming design matrix for predicting deposits
     nbasis=40;
     Xmat_design_lin_dep=[ones(n,1) geo_data space_scaled_data];
     nvar_dep_lin=size(Xmat_design_lin_dep,2);
     nvar_dep=nvar_dep_lin+nbasis;
     XX_dep_expand=zeros(n,nbasis);
     XX_dep_obs_unique=cell(num_era,1);
     XX_dep_obs_expand=cell(num_era,1);
     XX_dep_miss_unique=cell(num_era,1);
     XX_dep_miss_expand=cell(num_era,1);
     zmat_expand=zeros(n,nvar_dep);
     nvar_dep_lin=size(Xmat_design_lin_dep,2);
     space_unique_obs=cell(num_era,1);
     space_unique_miss=cell(num_era,1);
     zmat_obs_expand=cell(num_era,1);
     zmat_miss_expand=cell(num_era,1);
     cov_invvar_miss_obs=cell(num_era,1);
     zdashz_obs=cell(num_era,1);
     zzinv_obs=cell(num_era,1);
     dep_data_obs=cell(num_era,1);
     dep_data_miss=cell(num_era,1);
     np_pred_dep=nbasis+nvar_dep_lin; 
     ia_space_obs_unique=cell(num_era,1);
     ic_space_obs_unique=cell(num_era,1);
     ia_space_miss_unique=cell(num_era,1);
     ic_space_miss_unique=cell(num_era,1);
     for k=1:num_era
        dep_data_obs{k}=deposit_data(dep_obs_index{k},:);
        dep_data_miss{k}=deposit_data(dep_miss_index{k},:);
        [space_unique_obs{k},ia_space_obs_unique{k},ic_space_obs_unique{k}]=unique(space_scaled_data(dep_obs_index{k},1:2),'rows','stable');
        n_dep_obs_unique=length(space_unique_obs{k});
        [space_unique_miss{k},ia_space_miss_unique{k},ic_space_miss_unique{k}]=unique(space_scaled_data(dep_miss_index{k},1:2),'rows','stable');
         n_dep_miss_unique=length(space_unique_miss{k});
 %Generating covariance matrices for deposit prediction
        [omega_obs]=thinplate_basis(space_unique_obs{k}(:,1),space_unique_obs{k}(:,2),space_unique_obs{k}(:,1),space_unique_obs{k}(:,2));
        [omega_cov_obs_miss]=thinplate_basis(space_unique_miss{k}(:,1),space_unique_miss{k}(:,2),space_unique_obs{k}(:,1),space_unique_obs{k}(:,2));
        [omega_miss]=thinplate_basis(space_unique_miss{k}(:,1),space_unique_miss{k}(:,2),space_unique_miss{k}(:,1),space_unique_miss{k}(:,2));
         cov_invvar_miss_obs{k}(:,:)=omega_cov_obs_miss/omega_obs;
        %Representing the Kernal as a linear combination of basis functions
        %for observed values
         [Q, D]=eig(omega_obs);
         [d,ind] = sort(diag(D));
         D=D(ind,ind);
         Q=Q(:,ind);
         XX_dep_obs_unique{k}=Q*D^.5;
         XX_dep_obs_unique{k}=XX_dep_obs_unique{k}(:,n_dep_obs_unique-nbasis+1:n_dep_obs_unique);
         XX_dep_obs_expand{k}=XX_dep_obs_unique{k}(ic_space_obs_unique{k},:);
         zmat_obs_expand{k}=[Xmat_design_lin_dep(dep_obs_index{k},:) XX_dep_obs_expand{k}(:,1:nbasis)];
         zdashz_obs{k}(:,:)=zmat_obs_expand{k}'*zmat_obs_expand{k};
         zzinv_obs{k}(:,:)=inv(zdashz_obs{k}); 
         %Representing the Kernal as a linear combination of basis functions for missings value
        [Q, D]=eig(omega_miss);
        [d,ind] = sort(diag(D));
         D=D(ind,ind);
         Q=Q(:,ind);
         XX_dep_miss_unique{k}=Q*D^.5;
         XX_dep_miss_unique{k}=XX_dep_miss_unique{k}(:,n_dep_miss_unique-nbasis+1:n_dep_miss_unique);
         XX_dep_miss_expand{k}=XX_dep_miss_unique{k}(ic_space_miss_unique{k},:);
         XX_dep_expand(dep_miss_index{k},:)=XX_dep_miss_expand{k};
         XX_dep_expand(dep_obs_index{k},:)=XX_dep_obs_expand{k};
         zmat_miss_expand{k}(:,:)=[Xmat_design_lin_dep(dep_miss_index{k},:) XX_dep_miss_expand{k}(:,1:nbasis)];
         zmat_expand(dep_miss_index{k},:)=zmat_miss_expand{k};
         zmat_expand(dep_obs_index{k},:)=zmat_obs_expand{k};
     end
     %plot(Xmat_precip_obs_expand*xdashx_inv_exp*Xmat_precip_obs_expand'*y_precip_obs, y_precip_obs,'*')
%      ngrid=50;
%      nrep=1;
%      lat_grid=repmat(linspace(1/ngrid,1,ngrid)',nrep,1);
%      [lat_unique_grid,ia_grid,ic_grid]=unique(lat_grid);
%      [omega_temp]=cubic_basis(lat_unique_grid,lat_unique_grid);
%      omega_precip_lat_grid=omega_temp;
%      [cov_grid_obs]=cubic_basis(lat_unique_grid,lat_unique_obs);
%      cov_lat_invvar_precip_unique=cov_grid_obs/omega_lat_precip_obs;
%      cov_lat_invvar_precip_expand=cov_lat_invvar_precip_unique(ic_grid,:);
%      Xmat_precip_miss_lin=[ones(nrep*ngrid,1) lat_grid];
%      Xmat_precip_lin_exp=[Xmat_precip_obs_lin_exp;Xmat_precip_miss_lin];
%      nvar_precip_lin=size(Xmat_precip_obs_lin_exp,2);%design matrix for predicting precipitation which excludes spatial component
%      nvar_precip=nvar_precip_lin+nbasis_lat_elev;
   
     c_beta=ntrain;
     Sigsq=ones(nloop+nwarmup,1);
     Sigsq(1)=var(y_precip_obs);
     Sigsq_inv=diag(1/Sigsq(1));
     Tau_Beta=10000*ones(nloop+nwarmup,1);
     sigsq_prior_a=1;
     sigsq_prior_b=1;
     tau_prior_a=1;
     tau_prior_b=1;
     precip_fit_obs=zeros(nlat_elev_obs_unique,nloop+nwarmup+1);
     precip_fit_obs_expand=zeros(ntrain,nloop+nwarmup+1);
     precip_fit_obs_unique=zeros(nlat_elev_obs_unique,nloop+nwarmup+1);
     precip_fit_miss=zeros(nlat_elev_miss_unique,nloop+nwarmup+1);
     precip_fit_miss_expand=cell(num_era,1);
     precip_fit_miss_lat_unique=zeros(nlat_elev_miss_unique,nloop+nwarmup+1);
     Alpha_star=zeros(nvar_dep,num_era,num_dep,nloop+nwarmup);
     Tau_Alpha=ones(nvar_dep,num_era,num_dep);
     tau_betas=ones(nloop+nwarmup,1);
     c1=ones(num_era,1);
     tau_spline=10000*ones(num_era,num_dep,nloop+nwarmup);
     for k=1:num_era
        c1(k)=n_era(k);
        for j=1:num_dep
            Tau_Alpha(:,k,j)=[c1(k)*ones(nvar_dep_lin,1); tau_spline(k,j,1)*ones(nbasis,1)];
                %w_obs{k}(:,j)=deposit_data(dep_obs_index{k},j);%Constrained normals 
        end
     end
     Beta_star=zeros(nvar_precip,nloop+nwarmup+1);
     beta_hat=xdashx_inv_expand*Xmat_precip_obs_expand'*y_precip_obs;
     Beta_star(:,1)=beta_hat;
        
     fit_obs_expand=cell(num_era,1);
     fit_obs_unique=cell(num_era,1);
     fit_miss_expand=cell(num_era,1);
     fit_miss_unique=cell(num_era,1);
     y_dep_miss=cell(num_era,1);
     w_obs=cell(num_era,1);
     prob_dep_obs=cell(num_era,1);
     prob_dep_miss=cell(num_era,1);
     plot(lat_scaled_obs,Xmat_precip_obs_expand*Beta_star(:,1),'*')
     figure
     plot(elev_scaled_obs,Xmat_precip_obs_expand*Beta_star(:,1),'*')
     for p=1:nloop+nwarmup
         for k=1:num_era
            dep_data_obs_ind_all=deposit_data(dep_obs_index{k},:);
            for j=1:num_dep
                    %Generating constrained Gaussians
                    one_index=find(dep_data_obs_ind_all(:,j)==1);
                    zero_index=find(dep_data_obs_ind_all(:,j)==0);
                    uplim=zeros(nobs_dep_index(k),1);
                    uplim(one_index)=1000;
                    lowlim=zeros(nobs_dep_index(k),1);
                    lowlim(zero_index)=-1000;
                    w_mean=zmat_obs_expand{k}*Alpha_star(:,k,j,p);
                    w_var=ones(nobs_dep_index(k),1);
                    u=rand(nobs_dep_index(k),1);
                    Fb=normcdf(uplim,w_mean,ones(nobs_dep_index(k),1));
                    Fa=normcdf(lowlim,w_mean,ones(nobs_dep_index(k),1));
                    uu=u.*(Fb-Fa)+Fa;
                    w_obs{k}(:,j)=norminv(uu,w_mean,ones(nobs_dep_index(k),1));
      %Drawing the expected value fit
                    prior_alpha_star_mean=zeros(nvar_dep,1);
                    prior_alpha_star_prec=diag(1./Tau_Alpha(:,k,j));
                    like_alpha_star_prec=zdashz_obs{k};
                    like_alpha_star_mean=zzinv_obs{k}*zmat_obs_expand{k}'*w_obs{k}(:,j);
                    post_alpha_star_prec=prior_alpha_star_prec+like_alpha_star_prec;
                    post_alpha_star_var=post_alpha_star_prec\eye(nvar_dep);
                    post_alpha_star_mean=post_alpha_star_var*(like_alpha_star_prec*like_alpha_star_mean+prior_alpha_star_prec*prior_alpha_star_mean);
                    post_alpha_star_var=chol(post_alpha_star_var)'*chol(post_alpha_star_var);
                    Alpha_star(:,k,j,p+1)=mvnrnd(post_alpha_star_mean,post_alpha_star_var)';
                    fit_obs_expand{k}(:,j,p+1)=zmat_obs_expand{k}*Alpha_star(:,k,j,p+1);%smooth fit at observed x
                    %Fits of argument of link function
                    temp=cov_invvar_miss_obs{k}*(XX_dep_obs_unique{k}*Alpha_star(nvar_dep_lin+1:end,k,j,p+1));
                    fit_miss_expand{k}(:,j,p+1)=zmat_miss_expand{k}(:,1:nvar_dep_lin)*Alpha_star(1:nvar_dep_lin,k,j,p+1)+temp(ic_space_miss_unique{k});
                   %Fits of probabilities
                    prob_dep_obs{k}(:,j,p+1)=normcdf(fit_obs_expand{k}(:,j,p+1));
                    prob_dep_miss{k}(:,j,p+1)=normcdf(fit_miss_expand{k}(:,j,p+1));
                    uu=rand(nmiss_dep_index(k),1);
                    y_dep_miss{k}(:,j,p+1)=zeros(nmiss_dep_index(k),1);
                    y_dep_miss{k}(uu<prob_dep_miss{k}(:,j,p+1),j,p+1)=1;
                    Xmat_precip_expand(dep_miss_index{k},j+4)=y_dep_miss{k}(:,j,p+1);%assuming the deposit covariates are in cols 5:7
                    
           %Drawing tausq for deposits
                    tau_alpha_as=(nvar_dep-nvar_dep_lin-1)/2+tau_prior_a;
                    tau_alpha_bs=sum(Alpha_star(nvar_dep_lin+1:nvar_dep,k,j,p+1).^2)/2+tau_prior_b;
                    tau_spline(k,j,p+1)=1/gamrnd(tau_alpha_as,1./tau_alpha_bs);
                    Tau_Alpha(:,k,j)=[c1(k)*ones(nvar_dep_lin,1); tau_spline(k,j,p+1)*ones(nbasis,1)];
            end
            Xmat_precip_miss_lin_expand{k}=Xmat_precip_expand(era_index{k},1:nvar_precip_lin);
         end
         xdashx_expand=Xmat_precip_expand(obs_index,:)'*Xmat_precip_expand(obs_index,:);
         xdashx_inv_expand=eye(nvar_precip)/ xdashx_expand;
         prior_beta_star_mean=zeros(nvar_precip,1);
         prior_beta_star_prec=diag([ones(nvar_precip_lin,1)./c_beta;ones(nbasis_lat_elev,1)./Tau_Beta(p)]);
         like_beta_star_prec=Sigsq_inv*xdashx_expand;
         like_beta_star_mean=xdashx_inv_expand'*Xmat_precip_expand(obs_index,:)'*y_precip_obs;
         post_beta_star_prec=prior_beta_star_prec+like_beta_star_prec;
         post_beta_star_var=post_beta_star_prec\eye(nvar_precip);
         post_beta_star_mean=post_beta_star_var*(like_beta_star_prec*like_beta_star_mean+prior_beta_star_prec*prior_beta_star_mean);
         post_beta_star_var=chol(post_beta_star_var)'*chol(post_beta_star_var);
         Beta_star(:,p+1)=mvnrnd(post_beta_star_mean,post_beta_star_var)';
         precip_fit_obs_expand(:,p+1)=Xmat_precip_expand(obs_index,:)*Beta_star(:,p+1);
         ydev_obs=y_precip_obs-precip_fit_obs_expand(:,p+1);
         for k=3:num_era
            temp=cov_lat_elev_invvar_precip_unique{k}*(XX_lat_elev_precip_obs_unique*Beta_star(nvar_precip_lin+1:end,p+1));
            precip_fit_miss_expand{k}(:,p+1)=Xmat_precip_miss_lin_expand{k}*Beta_star(1:nvar_precip_lin,p+1)+temp(ic_lat_elev_miss_ind{k});
         end
        
        % y_precip_fit_miss_unique(:,p+1)=Xmat_precip_miss_lin_unique*Beta_star(1:nvar_precip_lin,p+1)+...
        % cov_lat_invvar_precip_unique*(XX_lat_precip_obs_unique*Beta_star(nvar_precip_lin+1:end,p+1));    
        %y_precip_fit_miss_expand(:,p+1)=y_precip_fit_miss_unique(ic_lat_miss,p+1);
         
         %drawing sigmasq
         sigsq_gampar_a=(ntrain+sigsq_prior_a)/2;
         sigsq_gampar_b=sigsq_prior_b+sum(ydev_obs.^2)/2;
         Sigsq(p+1)=1./gamrnd(sigsq_gampar_a,1./sigsq_gampar_b);
         Sigsq_inv=diag(1/Sigsq(p+1)); 
         %Drawing tausq_beta 
         tau_beta_ab=(nbasis_lat_elev-1)/2+tau_prior_a;
            tau_beta_bb=sum(Beta_star(nvar_precip_lin+1:nvar_precip,p+1).^2)/2+tau_prior_b;
            Tau_Beta(p+1)=1/gamrnd(tau_beta_ab,1./tau_beta_bb);
     end
    fit_all_med=zeros(n,num_dep);
    fit_all_lcl=zeros(n,num_dep);
    fit_all_ucl=zeros(n,num_dep);
    fit_obs_med=cell(num_era,1);
    fit_miss_med=cell(num_era,1);
    fit_obs_lcl=cell(num_era,1);
    fit_obs_ucl=cell(num_era,1);
    fit_miss_lcl=cell(num_era,1);
    fit_miss_ucl=cell(num_era,1);
    classify_obs=cell(num_era,1);
    classify_miss=cell(num_era,1);
    classify_all=cell(num_era,1);
    misclassify_rate=zeros(num_era,num_dep);
    for k=1:num_era
         for j=1:num_dep
            fit_obs_med{k}(:,j)=mean(prob_dep_obs{k}(:,j,nwarmup:nloop+nwarmup),3);
            fit_miss_med{k}(:,j)=mean(prob_dep_miss{k}(:,j,nwarmup:nloop+nwarmup),3);
            temp=squeeze(prob_dep_obs{k}(:,j,nwarmup:nloop+nwarmup));
            fit_obs_lcl{k}(:,j)=quantile(temp',0.05);
            fit_obs_ucl{k}(:,j)=quantile(temp',0.95);
            temp=squeeze(prob_dep_miss{k}(:,j,nwarmup:nloop+nwarmup));
            fit_miss_lcl{k}(:,j)=quantile(temp',0.05);
            fit_miss_ucl{k}(:,j)=quantile(temp',0.95);
            fit_all_med(dep_obs_index{k},j)=fit_obs_med{k}(:,j);
            fit_all_med(dep_miss_index{k},j)=fit_miss_med{k}(:,j);
            fit_all_lcl(dep_obs_index{k},j)=fit_obs_lcl{k}(:,j);
            fit_all_ucl(dep_obs_index{k},j)=fit_obs_ucl{k}(:,j);
            fit_all_lcl(dep_miss_index{k},j)=fit_miss_lcl{k}(:,j);
            fit_all_ucl(dep_miss_index{k},j)=fit_miss_ucl{k}(:,j);  
            classify_obs{k}(fit_obs_med{k}(:,j)>0.5,j)=1;
            classify_miss{k}(fit_miss_med{k}(:,j)>0.5,j)=1;
            classify_all{k}(dep_obs_index{k},j)=classify_obs{k}(:,j);
            classify_all{k}(dep_miss_index{k},j)=classify_miss{k}(:,j);
            misclassify_rate(k,j)=length(find(deposit_data(dep_obs_index{k},j)-classify_obs{k}(:,j)~=0))/nobs_dep_index(k);
         end
    end
    precip_fit_hat_miss_expand=cell(num_era,1);
    precip_lcl_hat_miss_expand=cell(num_era,1);
    precip_ucl_hat_miss_expand=cell(num_era,1);
    for k=3:num_era
        precip_fit_hat_miss_expand{k}=mean(precip_fit_miss_expand{k}(:,nwarmup:nloop+nwarmup),2);
        precip_lcl_hat_miss_expand{k}=quantile(precip_fit_miss_expand{k}(:,nwarmup:nloop+nwarmup),0.05,2);
        precip_ucl_hat_miss_expand{k}=quantile(precip_fit_miss_expand{k}(:,nwarmup:nloop+nwarmup),0.95,2);
    end
   
     precip_fit_hat_obs_expand=mean(precip_fit_obs_expand(:,nwarmup:nloop+nwarmup),2);
    precip_lcl_hat_obs_expand=quantile(precip_fit_obs_expand(:,nwarmup:nloop+nwarmup),0.05,2);
    precip_ucl_hat_obs_expand=quantile(precip_fit_obs_expand(:,nwarmup:nloop+nwarmup),0.95,2);
     figure
     hold
     plot(lat_scaled_obs,y_precip_obs.^(10/3),'*')
     plot(lat_scaled_obs,precip_fit_hat_obs_expand.^(10/3),'*')
      figure
     hold
      plot(elev_scaled_obs,y_precip_obs,'*')
     plot(elev_scaled_obs,precip_fit_hat_obs_expand,'*')
%      figure
%      for l=1:2
%         for k=3:floor(num_era/2)+2
%             if k+(l-1)*floor(num_era/2)<=num_era
%                 subplot(floor(num_era/2),2,k-2+(l-1)*floor(num_era/2))
%                 plot(lat_miss{k+(l-1)*floor(num_era/2)},precip_fit_hat_miss_expand(era_index{k+(l-1)*floor(num_era/2)}).^(10/3),'*')
%             end
%         end
%      end
     for k=3:num_era
         figure
        plot(lat_era{k},precip_fit_hat_miss_expand{k}.^(10/3),'*');
     end
    
    precip_fit_hat_miss=[];
    precip_lcl_hat_miss=[];
    precip_ucl_hat_miss=[];
    for k=3:num_era
    precip_fit_hat_miss=[precip_fit_hat_miss; precip_fit_hat_miss_expand{k}];
    precip_lcl_hat_miss=[precip_lcl_hat_miss; precip_lcl_hat_miss_expand{k}];
    precip_ucl_hat_miss=[precip_ucl_hat_miss; precip_ucl_hat_miss_expand{k}];
    end
    
    precip_fit_hat=[precip_fit_hat_obs_expand;precip_fit_hat_miss];
    precip_lcl_hat=[precip_lcl_hat_obs_expand;precip_lcl_hat_miss];
    precip_ucl_hat=[precip_ucl_hat_obs_expand;precip_ucl_hat_miss];
    y_precip_miss=-999*ones(ntest,1);
    y_precip_all=[y_precip_obs.^(10/3);y_precip_miss];
    results_all=[space_data(:,1), space_data(:,2), y_precip_all precip_fit_hat.^(10/3), precip_lcl_hat.^(10/3),precip_ucl_hat.^(10/3),fit_all_med, fit_all_lcl, fit_all_ucl, deposit_data, geo_data era_id];
    csvwrite('resultstrain_testall.csv',results_all);
    figure
    plot(y_precip_obs,precip_fit_hat_obs_expand,'*')
    
    resid_obs=y_precip_obs-precip_fit_hat_obs_expand;
    rmse_obs=mean((y_precip_obs-precip_fit_hat_obs_expand).^2);
    
    hh=figure
    for l=1:2
        for i=1:floor(nvar_precip_lin/2)
            if i+(l-1)*floor(nvar_precip_lin/2)+1<nvar_precip_lin
                subplot(floor(nvar_precip_lin/2),2,i+(l-1)*floor(nvar_precip_lin/2))
                histfit(squeeze(Beta_star(i+(l-1)*floor(nvar_precip_lin/2)+1,nwarmup:nloop+nwarmup)),30)
                ylabel({'p(\beta|Y)'},'FontWeight','bold','FontSize',14);
                xlabel({'\beta'},'FontWeight','bold','FontSize',14);
                title([predict_names{i+(l-1)*floor(nvar_precip_lin/2)},'','Rainfall','',int2str(j)]);
                hold on;
                line([0, 0], ylim, 'LineWidth', 2, 'Color', 'r');

            end
        end
    end 