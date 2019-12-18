     clear
     state=sum(100*clock);
     rand('state', state);
     randn('state', state);
     nloop=5000;
     nwarmup=5000;
     %Input Miocdene data
     deposit_data=readtable('deposit_input_all_eras_nov4.csv');% Table of 5 columns (1)lat (2)lon	(3)coal	(4)evap	(5)glacial,
     deposit_var_names=deposit_data.Properties.VariableNames;
     deposit_index=[3 4 5];%location of columns of coal, evaporite and glcial missing values are coded -999 .
     ndep=length(deposit_index);
     precip_predict_names=deposit_var_names(deposit_index);
     era_index=6;
     era_data=deposit_data(:,era_index);
     era_data=table2array(era_data);
     eras=unique(era_data,'stable');
     num_era=length(eras);

     deposit_data=deposit_data(:,deposit_index);
     deposit_data=table2array(deposit_data);
     n=length(deposit_data);
     dep_miss_ind=zeros(n,1);
     dep_miss_ind(deposit_data(:,1)==-999)=1;
     era_index=cell(num_era,1);
     dep_obs_index=cell(num_era,1);
     dep_miss_index=cell(num_era,1);
     nobs_dep_index=zeros(num_era,1);
     nmiss_dep_index=zeros(num_era,1);
     n_era=zeros(num_era,1);
     for k=1:num_era
         era_index{k}=find(era_data==eras(k));
         n_era(k)=length(era_index{k});
         dep_obs_index{k}=find(era_data==eras(k) & dep_miss_ind==0);
         dep_miss_index{k}=find(era_data==eras(k) & dep_miss_ind==1);
         nobs_dep_index(k)=length(dep_obs_index{k});
         nmiss_dep_index(k)=length(dep_miss_index{k});
     end
     geo_data=readtable('geo_input_all_eras_nov4.csv');%of the form region_id	lat	lon	Elev	Sqrt_Elev	D2S	SQRT_D2S	Sin	Cos
     geo_var_names=geo_data.Properties.VariableNames;
     geo_data=table2array(geo_data);
     %deposit_data_train=deposit_data_train(:,deposit_index);%deposit data for predicting precipitaion
     geo_index=[4 6 8 9];%selecting which geographical variables to use, for precipitation prediction in this case elevation D2S, sin and cos
     precip_predict_names=[precip_predict_names,geo_var_names(geo_index)];
     space_index=[2 3];
     space_data=geo_data(:,space_index);
     geo_data=geo_data(:,geo_index);
     geo_var_names=geo_var_names([geo_index,space_index]);
     %Spatial co-oerds
     lat=space_data(:,1);
     lon=space_data(:,2);
     %Scaling
     geo_data=(geo_data-ones(n,1).*min(geo_data))./(ones(n,1).*range(geo_data));%Scaling
     space_data=(space_data-ones(n,1).*min(space_data))./(ones(n,1).*range(space_data));%Scaling
     space_data=[space_data space_data(:,1).*space_data(:,2)];
     geo_var_names=[geo_var_names,'Space Interaction'];
     Xmat_design_dep=[ones(n,1) geo_data space_data];
     nvar_dep_lin=size(Xmat_design_dep,2);

%Logistic Regression

    %  dep_miss_index=find(deposit_data(:,1)==-999);%missing deposit index for training set
    %  dep_obs_index=find(deposit_data(:,1)~=-999);%observed deposit index for training set
    %  n_dep_obs=length(dep_obs_index);%Number of obs deposit
    %  n_dep_miss=length(dep_miss_index);%Number of m iss deposit
     %y_dep=deposit_data;%setting deposit as the response variabl
    %  y_dep_obs=deposit_data(dep_obs_index);%these are the observed values
    %  y_dep_miss=deposit_data(dep_miss_index);%these are the missing values
    %  
     %Setting up GP Prior and forming design matrix for predicting deposits

     zmat_obs=cell(num_era,1);
     zmat_miss=cell(num_era,1);
     cov_invvar_miss_obs=cell(num_era,1);
     zdashz_obs=cell(num_era,1);
     zzinv_obs=cell(num_era,1);
     dep_data_obs_ind=cell(num_era,1);
     dep_data_miss_ind=cell(num_era,1);
     nbasis=40;
     np_pred_dep=nbasis+nvar_dep_lin; 
     for k=1:num_era
        dep_data_obs_ind{k}=deposit_data(dep_obs_index{k});
        dep_data_miss_ind{k}=deposit_data(dep_miss_index{k});
        lat_obs=space_data(dep_obs_index{k},1);
        lon_obs=space_data(dep_obs_index{k},2);
        lat_miss=space_data(dep_miss_index{k},1);
        lon_miss=space_data(dep_miss_index{k},2);
 %Generating covariance matrices
        [omega_obs]=thinplate_basis(lat_obs,lon_obs,lat_obs,lon_obs);
        %Representing the Kernal as a linear combination of basis functions
        [Q, D]=eig(omega_obs);
         XX_obs=Q*D^.5;
         zmat_obs{k}(:,:)=[Xmat_design_dep(dep_obs_index{k},:) XX_obs(:,1:nbasis)];
         zdashz_obs{k}(:,:)=zmat_obs{k}'*zmat_obs{k};
         zzinv_obs{k}(:,:)=inv(zdashz_obs{k}); 
         if nmiss_dep_index(k)>1
             [omega_cov_obs_miss]=thinplate_basis(lat_obs,lon_obs,lat_miss,lon_miss);
             [omega_miss]=thinplate_basis(lat_miss,lon_miss,lat_miss,lon_miss);
              cov_invvar_miss_obs{k}(:,:)=omega_cov_obs_miss'/omega_obs;
              [Q, D]=eig(omega_miss);
              XX_miss=Q*D^.5;
              zmat_miss{k}(:,:)=[Xmat_design_dep(dep_miss_index{k},:) XX_miss(:,1:nbasis)];
         end
     end
%      figure
%      plot3(lat_miss,lon_miss,XX_miss(:,1),'*')
%      for j=1:3
%      y=deposit_data(dep_obs_index{1},j);
%      Xmat=zmat_obs{1}(:,1:end);
%      Xmat_miss=zmat_miss{1}(:,1:end);
%     [logitCoef,dev] = glmfit(Xmat,y,'binomial','probit');
%     logitFit(:,j) = glmval(logitCoef,Xmat_miss,'probit');
%      b = glmfit(Xmat,y,'binomial','link','probit');
%      logitFit2(:,j)=glmval(b,Xmat_miss,'probit');
%      end
%      [logitFit2 logitFit]
     %Reading in precipitation data
     precip_obs=readtable('precipinput_14Ma_v2_38&39.csv');% Table of 3 columns lat lon and precip
     precip_obs=table2array(precip_obs);
     y_index=3;%column location of preciptation. No missing precipitation results for the training data set  
     y_precip_obs=precip_obs(:,y_index).^0.3;% taking the cube root
     ntrain=size(precip_obs,1);% using the first two eras as training
     ntest=n-ntrain;
     Xmat_precip=[ones(n,1) deposit_data geo_data abs(lat) lat.^2];
     nvar_precip=size(Xmat_precip,2);%design matrix for predicting precipitation which excludes spatial component
     precip_predict_names=[precip_predict_names,"lat","lat.^2"];
     obs_precip_index=1:ntrain;
     miss_precip_index=1:ntest+ntrain ;
     sigsq_prior_a=1;
     sigsq_prior_b=1;
     tau_prior_a=1;
     tau_prior_b=1;
    Beta_star=zeros(nvar_precip,nwarmup+nloop);
    y_precip_fit=zeros(ntrain+ntest,nwarmup+nloop);
    tau_spline=10000*ones(num_era,ndep,nloop+nwarmup);
    %scale=var(y_obs)/4;
    %create storage space 

    w_obs=cell(num_era,1);%constrained normal
    prob_dep_obs=cell(num_era,1);%probability of dep
    prob_dep_miss=cell(num_era,1);%probability of dep
    y_dep_miss=cell(num_era,1);
    fit_obs=cell(num_era,1);
    fit_miss=cell(num_era,1);
    fit_obs_med=cell(num_era,1);
    fit_miss_med=cell(num_era,1);
    fit_obs_lcl=cell(num_era,1);
    fit_miss_lcl=cell(num_era,1);
    fit_obs_ucl=cell(num_era,1);
    fit_miss_ucl=cell(num_era,1);
    classify_obs=cell(num_era,1);
    classify_miss=cell(num_era,1);
    classify_all=cell(num_era,1);
    misclassify_rate=zeros(num_era,ndep);
    Alpha_star=zeros(np_pred_dep,num_era,ndep,nloop+nwarmup);
    Tau_Alpha=ones(np_pred_dep,num_era,ndep,nloop+nwarmup);
    c1=ones(num_era,1);
    for k=1:num_era
        c1(k)=nobs_dep_index(k);
        classify_obs{k}=zeros(nobs_dep_index(k),ndep);
        classify_miss{k}=zeros(nmiss_dep_index(k),ndep);
        classify_all{k}=zeros(n_era(k),ndep);
        prob_dep_obs{k}=zeros(nobs_dep_index(k),ndep,nwarmup+nloop);%fitted values
        prob_dep_miss{k}=zeros(nmiss_dep_index(k),ndep,nwarmup+nloop); %fitted values 
        y_dep_miss{k}=zeros(nmiss_dep_index(k),ndep,nwarmup+nloop); 
        fit_obs{k}=zeros(nobs_dep_index(k),ndep,nwarmup+nloop);
        fit_miss{k}=zeros(nmiss_dep_index(k),ndep,nwarmup+nloop);
         for j=1:ndep
            Tau_Alpha(:,k,j)=[c1(k)*ones(nvar_dep_lin,1); tau_spline(k,j,1)*ones(nbasis,1)];
            w_obs{k}(:,j)=deposit_data(dep_obs_index{k},j);%Constrained normals 
        end
    end


     %initialise parameters

    % kappa=ones(nobs,1);
    % Sigsq_inv=diag(kappa./scale);
    % Sigsq=diag(scale./kappa);
    % 
    % Beta_star=zeros(nvar,1);
    % zmat_obs=Xlin_obs;
    % zmat_miss=Xlin_miss;
    % zdashz_obs=Xlin_obs'*Xlin_obs;
    % zzinv_obs=inv(zdashz_obs);
    nu=100;
    Sigsq=ones(nloop+nwarmup,1);
    Sigsq_inv=1/Sigsq(1);
    c_beta=n;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     GIBBS LOOP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic

        %Drawing the unobserved deposits by drawing constrrained normals
        %First for training data
         
      for p=1:nloop+nwarmup
         if(mod(p,100)==0)
                    p;k
                 toc
         end
         for k=1:num_era
            %if (nmiss_dep_index(k)>1)
                dep_data_obs_ind_all=deposit_data(dep_obs_index{k},:);
                for j=1:ndep
                    one_index=find(dep_data_obs_ind_all(:,j)==1);
                    zero_index=find(dep_data_obs_ind_all(:,j)==0);
                    uplim=zeros(nobs_dep_index(k),1);
                    uplim(one_index)=1000;
                    lowlim=zeros(nobs_dep_index(k),1);
                    lowlim(zero_index)=-1000;
                    w_mean=zmat_obs{k}*Alpha_star(:,k,j,p);
                    w_var=ones(nobs_dep_index(k),1);
                    u=rand(nobs_dep_index(k),1);
                    Fb=normcdf(uplim,w_mean,ones(nobs_dep_index(k),1));
                    Fa=normcdf(lowlim,w_mean,ones(nobs_dep_index(k),1));
                    uu=u.*(Fb-Fa)+Fa;
                    w_obs{k}(:,j)=norminv(uu,w_mean,ones(nobs_dep_index(k),1));
      %Drawing the expected value fit
                    prior_alpha_star_mean=zeros(np_pred_dep,1);
                    prior_alpha_star_prec=diag(1./Tau_Alpha(:,k,j));
                    like_alpha_star_prec=zdashz_obs{k};
                    like_alpha_star_mean=zzinv_obs{k}*zmat_obs{k}'*w_obs{k}(:,j);
                    post_alpha_star_prec=prior_alpha_star_prec+like_alpha_star_prec;
                    post_alpha_star_var=post_alpha_star_prec\eye(np_pred_dep);
                    post_alpha_star_mean=post_alpha_star_var*(like_alpha_star_prec*like_alpha_star_mean+prior_alpha_star_prec*prior_alpha_star_mean);
                    post_alpha_star_var=chol(post_alpha_star_var)'*chol(post_alpha_star_var);
                    Alpha_star(:,k,j,p+1)=mvnrnd(post_alpha_star_mean,post_alpha_star_var)';
                    fit_obs{k}(:,j,p+1)=zmat_obs{k}*Alpha_star(:,k,j,p+1);%smooth fit at observed x
                   if (nmiss_dep_index(k)>0)
                        fit_miss{k}(:,j,p+1)=Xmat_design_dep(dep_miss_index{k},:)*Alpha_star(1:nvar_dep_lin,k,j,p+1)+cov_invvar_miss_obs{k}...
                        *(fit_obs{k}(:,j,p+1)-Xmat_design_dep(dep_obs_index{k},:)*Alpha_star(1:nvar_dep_lin,k,j,p+1));
                   else
                       fit_miss{k}(:,j,p+1)=Xmat_design_dep(dep_miss_index{k},:)*Alpha_star(1:nvar_dep_lin,k,j,p+1);
                        
                   end
                    prob_dep_obs{k}(:,j,p+1)=normcdf(fit_obs{k}(:,j,p+1));
                    prob_dep_miss{k}(:,j,p+1)=normcdf(fit_miss{k}(:,j,p+1));
                    uu=rand(nmiss_dep_index(k),1);
                    y_dep_miss{k}(:,j,p+1)=zeros(nmiss_dep_index(k),1);
                    y_dep_miss{k}(uu<prob_dep_miss{k}(:,j,p+1),j,p+1)=1;
                    Xmat_precip(dep_miss_index{k},j+1)=y_dep_miss{k}(:,j,p+1);
                    
           %Drawing tausq 
                    tau_alpha_as=(np_pred_dep-nvar_dep_lin-1)/2+tau_prior_a;
                    tau_alpha_bs=sum(Alpha_star(nvar_dep_lin+1:np_pred_dep,k,j,p+1).^2)/2+tau_prior_b;
                    tau_spline(k,j,p+1)=1/gamrnd(tau_alpha_as,1./tau_alpha_bs);
                    Tau_Alpha(:,k,j)=[c1(k)*ones(nvar_dep_lin,1); tau_spline(k,j,p+1)*ones(nbasis,1)];
                end
            %end
         end
         xdashx=Xmat_precip(obs_precip_index,:)'*Xmat_precip(obs_precip_index,:);
         xdashx_inv=eye(nvar_precip)/ xdashx;
         prior_beta_star_mean=zeros(nvar_precip,1);
         prior_beta_star_prec=diag(ones(nvar_precip,1)./c_beta);
         like_beta_star_prec=Sigsq_inv*xdashx;
         like_beta_star_mean=xdashx_inv'*Xmat_precip(obs_precip_index,:)'*y_precip_obs;
         post_beta_star_prec=prior_beta_star_prec+like_beta_star_prec;
         post_beta_star_var=post_beta_star_prec\eye(nvar_precip);
         post_beta_star_mean=post_beta_star_var*(like_beta_star_prec*like_beta_star_mean+prior_beta_star_prec*prior_beta_star_mean);
         post_beta_star_var=chol(post_beta_star_var)'*chol(post_beta_star_var);
         Beta_star(:,p+1)=mvnrnd(post_beta_star_mean,post_beta_star_var)';
         y_precip_fit(:,p+1)=Xmat_precip*Beta_star(:,p+1);
         ydev_obs=y_precip_obs-y_precip_fit(obs_precip_index,p+1);
         sigsq_gampar_a=(ntrain+sigsq_prior_a)/2;
         %drawing sigmnasq
        sigsq_gampar_b=sigsq_prior_b+sum(ydev_obs.^2)/2;
        Sigsq(p+1)=1./gamrnd(sigsq_gampar_a,1./sigsq_gampar_b);
        Sigsq_inv=1/ Sigsq(p+1);  
      end
%Figures  for deposit predictors
fit_all_med=zeros(n,ndep);
fit_all_lcl=zeros(n,ndep);
fit_all_ucl=zeros(n,ndep);
for k=1:num_era
         for j=1:ndep
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
            
            misclassify_rate(k,j)=length(find(deposit_data(dep_obs_index{k},j)-classify_obs{k}(:,j))~=0)/nobs_dep_index(k);
            if k==1
                h=figure
                        %subplot(2,ceil(nvar_dep_lin/2),2*floor((nvar_dep_lin+1)/2))
                for l=1:2
                    for i=1:ceil(nvar_dep_lin/2)
                        if i+(l-1)*ceil(nvar_dep_lin/2)+1<nvar_dep_lin+1
                            g = subplot(ceil(nvar_dep_lin/2),2,i+(l-1)*ceil(nvar_dep_lin/2));
                            histfit( squeeze(Alpha_star(i+(l-1)*ceil(nvar_dep_lin/2)+1,k,j,nwarmup:nloop+nwarmup)),30)
                             ylabel({'p(\alpha|Y)'},'FontWeight','bold','FontSize',14);
                            xlabel({'\alpha'},'FontWeight','bold','FontSize',14);
                            title([geo_var_names{i+(l-1)*ceil(nvar_dep_lin/2)},'','Deposit Type','',int2str(j)]);
                            hold on;
                            line([0, 0], ylim, 'LineWidth', 2, 'Color', 'r');
                            
                            
                            
                        end
                    end
                end
                %savefig(h,'int2str(k)'+'_Predictors_Deposit.fig') . 
                %bug here in saving plots - Sally pls see
                %saveas(gcf,'int2str(k)'+'_Predictors_Deposit.png')
            end
         end
        hh=figure
        for l=1:2
            for i=1:ceil(nvar_precip/2)
                if i+(l-1)*ceil(nvar_precip/2)+1<nvar_precip+1
                    subplot(ceil(nvar_precip/2),2,i+(l-1)*ceil(nvar_precip/2))
                    histfit(squeeze(Beta_star(i+(l-1)*ceil(nvar_precip/2)+1,nwarmup:nloop+nwarmup)),30)
                     ylabel({'p(\beta|Y)'},'FontWeight','bold','FontSize',14);
                    xlabel({'\beta'},'FontWeight','bold','FontSize',14);
                    title([precip_predict_names{i+(l-1)*ceil(nvar_precip/2)},'','Rainfall','',int2str(j)]);
                    hold on;
                    line([0, 0], ylim, 'LineWidth', 2, 'Color', 'r');
                    
                end
            end
        end
       %savefig(hh,"Predictors_Rain.fig") % bug here in saving plots 
    end
    
      %Write deposit predictions
     
     
    % %Graphs
     precip_fit_hat=mean(y_precip_fit(:,nwarmup:nloop+nwarmup),2);
     precip_fit_lcl=quantile(y_precip_fit(:,nwarmup:nloop+nwarmup)',0.05)';
     precip_fit_ucl=quantile(y_precip_fit(:,nwarmup:nloop+nwarmup)',0.95)';
     %rescaling the geographical data
     n_precip=length(y_precip_obs);
     y_precip_all=[y_precip_obs;-999*ones(n-n_precip,1)]
    results=[lat,lon, y_precip_all.^(10/3) precip_fit_hat.^(10/3), precip_fit_lcl.^(10/3),precip_fit_ucl.^(10/3),fit_all_med, fit_all_lcl, fit_all_ucl, deposit_data, geo_data, era_data];
     csvwrite('results_all.csv',results);
        figure
      plot(y_precip_obs,precip_fit_hat(obs_precip_index),'*')
      csvwrite('missclassification.csv',misclassify_rate)
%      saveas(gcf,"Fit_vs_Actual.png")  
         
    % % miss_fit_hat=mean(Y_miss_fit_all(:,nwarmup:nloop),2);
    % % low_cred_lim_obs=quantile(Y_obs_fit_all(:,nwarmup:nloop),0.05,2);
    % % up_cred_lim_obs=quantile(Y_obs_fit_all(:,nwarmup:nloop),0.95,2);
    % % low_cred_lim_miss=quantile(Y_miss_fit_all(:,nwarmup:nloop),0.05,2);
    % % up_cred_lim_miss=quantile(Y_miss_fit_all(:,nwarmup:nloop),0.95,2);
    % % results1=[miss_fit_hat low_cred_lim_miss up_cred_lim_miss precip_miss_coords era_data_miss];
    % % csvwrite('all_results',results1)
    % 
    % for j=1:num_era
    %     index=find(era_data_miss==eras(j));
    %     csvwrite(num2str(eras(j)),[miss_fit_hat(index) low_cred_lim_miss(index)...
    %         up_cred_lim_miss(index) precip_miss_coords(index,:)]);
    % end
    % figure
    % subplot(2,2,1)
    % hist(sqrt(Scale_all(nwarmup:nloop)),50);
    % title(['Distribition of Sacle']);
    % 
    % subplot(2,2,2)
    % plot(obs_fit_hat,y_obs,'*r');
    % % hold
    % % plot(miss_fit_hat,y_miss,'*b');
    % title('Predicted vs Actual, Observed in red, Missing in Blue')
    % lsline
    % subplot(2,2,3)
    % histfit(y_obs-obs_fit_hat,40,'tlocationscale')
    % title(['Histogram of Residuals with t_4 distribution in red'])
    % subplot(2,2,4)
    % boxplot(Beta_star_all(2:nvar,nwarmup:nloop)')%'labels','varnames(1:nvar-3)');
    % title(['Boxplot of Regression Coefficients'])
   
    %         if k+(j-1)*ceil(nvar/2)+1<nvar+1
    %         subplot(ceil(nvar/2),2,k+(j-1)*ceil(nvar/2))
    %         histfit(Beta_star_all(k+(j-1)*ceil(nvar/2)+1,nwarmup:nloop)',30)
    %         ylabel({'p(\beta|Y)'},'FontWeight','bold','FontSize',14);
    %         xlabel({'\beta'},'FontWeight','bold','FontSize',14);
    % 
    % % Create title
    %     title(depositvarnames{k+(j-1)*ceil(nvar/2)});
    %     hold on;
    %     line([0, 0], 
    
    %         end
    %     end
    % end
    % 

