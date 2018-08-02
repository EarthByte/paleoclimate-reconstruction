 clear
 state=sum(100*clock);
 rand('state', state);
 randn('state', state);
 nloop=5000;
 nwarmup=2000;
 load('datamissing.mat');
 load('var_names.mat');
 datamissing=table2array(datamissing);
 y_index=5;%
 ntot=size(datamissing,1);
 y=datamissing(:,y_index);
 deposit_xindex=[6];%columns in "data" which contain deposits; note coal =6
 X_deposit=datamissing(:,deposit_xindex);
 nmiss=sum(isnan(datamissing(:,deposit_xindex)));
 miss_index=find(isnan(datamissing(:,deposit_xindex))==1);
 obs_index=find(isnan(datamissing(:,deposit_xindex))==0);
 nobs=ntot-nmiss;
 nvar_dep=length(deposit_xindex);
 cont_xindex=[12];% column of continuous variabels other than lat and lon
 nvar_cont=length(cont_xindex);
 space_xindex=[9,10];%columns of lat and lon
 lin_index=[cont_xindex space_xindex];
 Xspace=datamissing(:,space_xindex);
 nvar=length(lin_index);
 cov_names=var_names(lin_index);
 Xlin=[ones(ntot,1),datamissing(:,lin_index)];
 Xlin(:,nvar_dep+2:nvar+1)=(Xlin(:,nvar_dep+2:nvar+1)-min(Xlin(:,nvar_dep+2:nvar+1)))./range(Xlin(:,nvar_dep+2:nvar+1));%standardizing
 %GP Kernel
 ngrid=60;
 lat=Xlin(:,nvar);
 lon=Xlin(:,nvar+1);
 lat_grid=repmat(linspace(min(lat),max(lat),ngrid)',ngrid,1);
 lon_grid=reshape(repmat(linspace(min(lon),max(lon),ngrid),ngrid,1),ngrid^2,1);
 zgrid=[ones(ngrid^2,1) lat_grid, lon_grid];
 gridx=Xlin(:,nvar);
 gridy=Xlin(:,nvar+1);
 non_linear=1;
 lat_obs=lat(obs_index);
 lon_obs=lon(obs_index);
 Xspace_obs=[lat_obs,lon_obs];
 Xlin_obs=Xlin(obs_index,:);
 Xlin_land=Xlin;
 Xlin_coal=[ones(nobs,1) Xspace_obs];
 coal_obs=Xlin_obs(:,2);
 
    if non_linear==1 
        [omega_coal]=thinplate_basis(lat_obs,lon_obs,lat_obs,lon_obs);
        [omega_land]=thinplate_basis(lat,lon,lat,lon);
        [omega_grid]=thinplate_basis(lat_grid,lon_grid,lat_grid,lon_grid);
        [omega_cov_coal_land]=thinplate_basis(lat_obs,lon_obs,lat,lon);
        [omega_cov_land_grid]=thinplate_basis(lat,lon,lat_grid,lon_grid);
        omega_cov_invvar_coal_land=omega_cov_coal_land'/omega_coal;
        omega_cov_invvar_land_grid=omega_cov_land_grid'/omega_land;

%omega_cov_grid_design=omega_data_grid'/omega;
%Representing the Kernal as a linear combination of basis functions
        [Q D]=eig(omega_land);
         XX_land=Q*D^.5;
        [Q D]=eig(omega_coal);
         XX_coal=Q*D^.5;
        nbasis=80;
    else
        nbasis=0;
        XX_land=[];
        XX_coal=[]; 
    end
    zmat_land=[Xlin_land XX_land(:,1:nbasis)];
    zmat_coal=[Xlin_coal XX_coal(:,1:nbasis)];
    np=size(zmat_land,2);
    np_coal=size(zmat_coal,2);
    nfixed=np-nbasis;
    nfixed_coal=np_coal-nbasis;
        %PUT IN SENSIBLE HYPERPARAMS
    tau_prior_a=-1;
    tau_prior_b=0;
%create storage space 
    Beta_star_all=zeros(np,nloop+nwarmup);
    Sigsq_all=zeros(1,nloop+nwarmup);
    Tau_Spline_all=zeros(1,nwarmup+nloop);
    Y_fit_all=zeros(ntot,nloop+nwarmup);
    log_likelihood_all=zeros(1,nloop+nwarmup);
    sigsq_gampar_a=zeros(nloop+nwarmup,1);
    sigsq_gampar_b=zeros(nloop+nwarmup,1);
     Beta_star_coal_all=zeros(np_coal,nloop+nwarmup);
    Sigsq_coal_all=zeros(1,nloop+nwarmup);
    Tau_Spline_coal_all=zeros(1,nwarmup+nloop);
    coal_fit_all=zeros(nobs,nloop+nwarmup);
    log_likelihood_coal_all=zeros(1,nloop+nwarmup);
    upl=zeros(ntot,nloop);
    lpl=zeros(ntot,nloop);
    ucl=zeros(ntot,nloop);
    lcl=zeros(ntot,nloop);
        %initialise parameters
    c1=ntot/4;
    tau_spline=10000;
    tau_spline_coal=10000;
    Sigsq=var(y)/4;
    Tau_Beta=[c1*Sigsq*ones(nfixed,1);tau_spline*ones(nbasis,1)];
    Tau_Beta_coal=[ones(nfixed_coal,1);tau_spline_coal*ones(nbasis,1)];
    Beta_star=zeros(np,1);
    Beta_star_coal=zeros(np_coal,1);
    zdashz_land=zmat_land'*zmat_land;
    zzinv_land=inv(zdashz_land);
    zdashz_coal=zmat_coal'*zmat_coal;
    zzinv_coal=inv(zdashz_coal);  
    var_y=ones(ntot,1);
    var_ftbar=ones(ntot,1);
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
           like_beta_star_prec=zdashz/Sigsq;
           like_beta_star_mean=zzinv*zmat'*y;
           post_beta_star_prec=prior_beta_star_prec+like_beta_star_prec;
           post_beta_star_var=post_beta_star_prec\eye(np);
           post_beta_star_mean=post_beta_star_var*(like_beta_star_prec*like_beta_star_mean+prior_beta_star_prec*prior_beta_star_mean);
           post_beta_star_var=chol(post_beta_star_var)'*chol(post_beta_star_var);
           Beta_star=mvnrnd(post_beta_star_mean,post_beta_star_var)';
           yfit=zmat*Beta_star;%smooth fit at observed x
           ydev=y-yfit;
           Beta_grid=[Beta_star(1) Beta_star(nvar) Beta_star(nvar+1)]';
           fit_grid=zgrid*Beta_grid+omega_cov_grid_design...
         *(zmat*Beta_star-Xlin*Beta_star(1:nvar+1));
           for t=1:nobs
               var_ftbar(t)=zmat(t,:)*post_beta_star_var*zmat(t,:)';
               var_y(t)=var_ftbar(t)+Sigsq;
           end
           upl(:,p)=yfit+2*sqrt(var_y);
           lpl(:,p)=yfit-2*sqrt(var_y);
           ucl(:,p)=yfit+2*sqrt(var_ftbar);
           lcl(:,p)=yfit-2*sqrt(var_ftbar);
      
           % Drawing Sigmasq 
           sigsq_gampar_b(p)=sum(ydev.^2)/2+sigsq_prior_b;
           sigsq_gampar_a(p)=nobs/2-1+sigsq_prior_a;
           Sigsq=1/gamrnd(sigsq_gampar_a(p),1/sigsq_gampar_b(p));
           
           %Drawing mean of random effects slope Note that
           %mean of spline is zero
           if non_linear==1
           tau_beta_as=(np-nfixed-1)/2+tau_prior_a;
           tau_beta_bs=sum(Beta_star(nfixed+1:np).^2)/2+tau_prior_b;
           tau_spline=1/gamrnd(tau_beta_as,1./tau_beta_bs);
           
           %tau_spline=10000
          
           Tau_Beta=[c1*Sigsq*ones(nfixed,1);tau_spline*ones(nbasis,1)]; 
           else
               Tau_Beta=[c1*Sigsq*ones(nfixed,1)];
           end
           Beta_star_all(:,p)=Beta_star;
           Sigsq_all(p)=Sigsq;
           Tau_Spline_all(p)=tau_spline;
           log_likelihood_all(p)=sum(log(normpdf(y,yfit,sqrt(Sigsq))));
           Y_fit_all(:,p)=zmat*Beta_star;
           fit_grid_all(:,p)=fit_grid;
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
fit_hat=mean(fit_grid_all(:,nwarmup:nloop),2)
mean_fit=mean(Y_fit_all(:,nwarmup:nloop),2)
        figure
        subplot(1,3,1)
        hist(sqrt(Sigsq_all(nwarmup:nloop)),50)
         title(['Distribition of RMSE'])
        
       % max_index=find(log_likelihood_all==max(log_likelihood_all(nwarmup)));
        subplot(1,3,2)
        boxplot(Beta_star_all(2:nvar_dep+nvar_cont+1,nwarmup:nloop)','Labels',cov_names(1:end-2));
        title(['Boxplot of Regression Coefficients'])
        xlabel(['Number of obs= ', num2str(nobs), '  Nonlinear=  ', num2str(non_linear)])
        yhat=mean(Y_fit_all(:,nwarmup+1:nloop),2);
        %yfit_max_like=Y_fit_all(:,max_index)  
        resid=y-yhat;
       subplot(1,3,3)
        histfit(resid,round(nobs/20),'tlocationscale')
        name=cov_names(1:nvar-2);
        title(['Residuals '])
 
     xx=histfit(resid,round(nobs/20),'tlocationscale');
     figure
    tri = delaunay(Xlin(:,nvar),Xlin(:,nvar+1));
    trisurf(tri, Xlin(:,nvar),Xlin(:,nvar+1), mean_fit);
% optional, could help make the plot look nicer
lat_data=(Xspace(:,1)-min(Xspace(:,1)))./(max(Xspace(:,1))-min(Xspace(:,1)));
lon_data=(Xspace(:,2)-min(Xspace(:,2)))./(max(Xspace(:,2))-min(Xspace(:,2)));
figure

 tempx_grid1=linspace(min(lat_data),max(lat_data),ngrid)';
 tempy_grid1=linspace(min(lon_data),max(lon_data),ngrid)';
 fit_hat1=reshape(fit_hat,ngrid,ngrid);
 figure
 surf(tempy_grid1,tempx_grid1,fit_hat1)
 title(['Fit on Grid assuming all covariates=0 '])
 hold
 plot3(lat_data,lon_data,mean_fit,'*')

