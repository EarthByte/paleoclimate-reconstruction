"""
Main Modul with scripts for runnning MCMC with Bayesian Linear reggression and Gaussian Process
Version 0.1
Author: Sebastian Haan
"""

from __future__ import division, print_function
from mpl_toolkits.mplot3d import *
import matplotlib.pylab as plt
from matplotlib import cm
import datetime as dt
import os
import sys
import emcee
import george # to be be replaced soon with manual GP calculation to gain additional speed
from george import kernels # replaced soon with more customizable kernels 
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
sns.set(font_scale=1.5)
#np.random.seed(3121)
from settings import *


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


class GPMC:
    """
    Class for combining Gaussian Process with Bayesian Linear Regression (BLR)
    For mcmc sampling the emcee package is used which is an implementation of
    affineinvariant ensemble sampler for Markov chain Monte Carlo (MCMC) proposed
    by Goodman & Weare (2010).
    Within MCMC loop the mean values of the BLR are forwarded to the Gaussian Process.
    """
    def __init__(self,outmcmc, split_traintest):
        """
        :param outmcmc: path to directory for output files, if it does not exists, a new directory will be created
        :param split_traintest: fraction of test data ranging from 0 (only training) to 0.5 (50/50 train/test)
        """
        self.y = []
        self.ndata = 0.
        self.y_std = 0.
        self.X_gp = []
        self.X_blr = []
        self.x_min = 0.
        self.x_max = 0.
        self.ndim_blr = 0
        self.ndim_gp = 2
        #other variables for calculation and output
        self.residual_blr = [] #stores residuals of BLR model for mean of GP calcuation
        self.niter = 1000
        self.nwalkers = 50
        #for spatial image data
        self.xv = []
        self.data_gp = []
        self.data_blr = []
        self.data = []
        self.model_gp = []
        self.model_blr = []
        self.model = []
        self.params_fit = []
        self.errors_fit = []
        self.create_data_bool = False
        self.outmcmc = outmcmc
        if not os.path.exists(outmcmc):
            os.makedirs(outmcmc)
        self.results_file = open(outmcmc+'mcmc_results.txt', "w")
        self.ymodel_mean = []
        self.ymodel_std  = []
        if (split_traintest > 0.) & (split_traintest <=0.5):
            self.y_test = []
            self.x_gp_test = []
            self.x_blr_test = []
            self.y_model_test = []
            self.ymodel_mean_test = []
            self.ymodel_std_test  = []
            self.residual_blr_test = []
        elif split_traintest == 0:
            print("Warning: only training data, no validation on test set.")
        elif split_traintest > 0.5:
            print("Test data must be less or equal to 0.5 times size of total data!")
            sys.exit(0)


    def load_data(self, y, X_gp, X_blr, y_test, X_gp_test, X_blr_test):
        """ Loads data in class variables
        :param y: train y data
        :param X_gp: train data with spatial center x,y coordinate
        :param X_blr: train data for linear regression features
        :param y_test: test y data
        :param X_gp_test: test data with spatial center x,y coordinate
        :param X_blr_test: test data for linear regression features (same number of features as train data required)
        :return: no return, saves output in class variables
        """
        # train data:
        self.y = y
        self.y_std = np.std(y)
        self.ndata = len(y)
        self.X_gp = X_gp # 2-dimensions required
        self.X_blr = X_blr # can be multiple dimensions
        #dist_matrix_dim1 = pairwise_distances(self.X_gp[:,0])
        #self.maxL_dim1 = np.log(min(dist_matrix_dim1))
        # test data:
        self.y_test = y_test
        self.X_gp_test = X_gp_test
        self.X_blr_test = X_blr_test
        self.x_min = np.min(self.X_gp)
        self.x_max = np.max(self.X_gp)
        self.ndim_blr = X_blr.shape[1]

    def out_data(self, y, X_gp, X_blr):
        """ saves normalized data in extra csv files
        :param y: y data
        :param X_gp: spatial center coordinates
        :param X_blr: features for X
        :return: no return
        """
        print('saving normalized input data')
        np.savetxt(self.outmcmc + 'inputdata_y.csv', y, delimiter=",", fmt='%s')
        np.savetxt(self.outmcmc + 'inputdata_X_gp.csv', X_gp, delimiter=",", fmt='%s')
        np.savetxt(self.outmcmc + 'inputdata_X_blr.csv', X_blr, delimiter=",", fmt='%s')

    def create_data(self, path, x_min, x_max, nsize, noise=0.1):
        """ Creates simulated data for model with 2 BLR features and a 2-dim spatial component for Gaussian Process
        The spatia component has a linear component in x and y direction plus an additional exponential component
        depending on the radial distance.

        :param x_min: lower boundary of spatial component
        :param x_max:  upper boundary of spatial component
        :param nsize: size of simulated dataset
        :param noise: fraction of noise, default 0.1
        :return: None, saves data in csv file
        """
        print("Create simulated crime data and spatial data...")
        #create 2D test data for image nsize * nsize
        sx = np.linspace(x_min, x_max, nsize)
        sy = np.linspace(x_min, x_max, nsize)
        xv, yv = np.meshgrid(sx, sy)
        radius1 = np.sqrt((xv - 4.) ** 2 + (yv - 6) ** 2)
        #radius2 = np.sqrt((xv + 6) ** 2 + (yv + 6) ** 2)
        #X = np.zeros((nsize * nsize, ndim_blr + 2))
        #X[:, 0] = xv.flatten()
        #X[:, 1] = yv.flatten()
        # Make BLR data
        data_spatial = 0.2 * yv + 0.3 *xv + 3 * np.exp(-radius1**2 / 4.)
        #data_spatial = 0.2 * yv + 5. * np.exp(-radius1 / 5.) + 0.5 * radius1 + 5. * np.exp(-radius2 / 5.)
        x1_blr = xv.copy() * 0.001 + np.random.rand(nsize, nsize)
        x2_blr = yv.copy() * 0.001 + np.random.rand(nsize, nsize)
        data_blr = 4.2 * x1_blr + 8.4 * x2_blr
       # X[:, 2] = x1_blr.flatten()
       # X[:, 3] = x2_blr.flatten()
        # data_blr_flat = 1.2* X[:,2]  + 2.4 * X[:,3] #* np.random.randn(len(X[:,2])) * 0.6
        #data = data_spatial + data_blr
        alpha_true = 5.
        #y = alpha_true + data_spatial.flatten() + data_blr.flatten() + noise * np.random.rand(nsize**2) 
        y = alpha_true + data_blr.flatten() + np.random.normal(0,noise,nsize**2)
        id = np.linspace(1, len(y), len(y)).astype(int)
        #Save simulated data as csv file:
        par_array = np.vstack([id, y, x1_blr.flatten(), x2_blr.flatten(), xv.flatten(), yv.flatten()]).T
        header = "region_id,"+target_name+",x1,x2,centroid_x,centroid_y"
        #header = "region_id,log_crime_rate,x1,x2,centroid_x,centroid_y"
        np.savetxt(path + 'simdata2D.csv', par_array, delimiter=",", header=header, fmt='%s', comments = '')


    def plot_2D(self, x, model_in,data_in):
        # experimental function, x and model must be 2dim arrays
        yi, xi = np.mgrid[self.x_min:self.x_max:100j, self.x_min:self.x_max:100j]
        #plt.plot(x[0], x[1], 'ko')
        plt.figure(1)
        plt.imshow(model_in, extent=[self.x_min, self.x_max, self.x_min, self.x_max], cmap='gist_earth')
        plt.title('Model')
        plt.colorbar()
        plt.figure(2)
        plt.imshow(data_in, extent=[self.x_min, self.x_max, self.x_min, self.x_max], cmap='gist_earth')
        plt.title('Data')
        plt.colorbar()
        #plt.show()

    def kernel_gp(self, kernelname = 'expsquared'):
        """ Kernel defininition
        Standard kernels provided with george, evtl replaced with manual customizable kernels
        User can change function to add new kernels or comninations thereof as much as required 
        :param kernelname: name of kernel, either 'expsquared', 'matern32', or 'rationalq'
        """
        # Kernel for spatial 2D Gaussian Process. Default Exponential Squared Kernel. Change accordingly below.
        # Kernels are initialised with weight and length (1 by default)
        if kernelname == 'expsquared':
            k0 = 0.1 * kernels.ExpSquaredKernel([0.1,0.1], ndim = 2)  # + kernels.WhiteKernel(1.,ndim=2)
        # other possible kernels as well as combinations:
        if kernelname == 'matern32':
            k0 = 1. * kernels.Matern32Kernel(1., ndim = 2)
        # k2 = 2.4 ** 2 * kernels.ExpSquaredKernel(90 ** 2) * kernels.ExpSine2Kernel(2.0 / 1.3 ** 2, 1.0)
        if kernelname == 'rationalq':
            k0 =  1. * kernels.RationalQuadraticKernel(0.78, 1.2, ndim=2)
        # k4 = kernels.WhiteKernel(1., ndim=2)
        return k0 # + k1 + k2 + ...

    def predict_blr(self, X_test, alpha, beta):
        # Sum of linear regression, alpha is a constant
        mu_blr = alpha  + np.sum(beta * X_test, axis=1)
        return mu_blr

    def lnprior_gp(self, p):
        if np.any((2.*np.log(self.y_std)-7. > p[0]) + (p[0] > 2.*np.log(self.y_std)+1.)): # amplitude (min-GP1=log(min-sigma**2), max-GP1=log(max-sigma**2))
            lnprior = -np.inf
        elif np.any((-4.6 > p[1]) + (p[1] > 0.)): # latitude length scale (min = log(1.5/145))
        #elif np.any((-4. > p[1]) + (p[1] > 0.)): # latitude length scale (min = log(2.5/140))
        #elif np.any((-3.3 > p[1]) + (p[1] > 0.)): # latitude length scale (min = log(5/135))
            lnprior = -np.inf
        elif np.any((-5.5 > p[2]) + (p[2] > 0.)): # longitude length scale (min = log(1.5/360))
        #elif np.any((-5. > p[2]) + (p[2] > 0.)): # longitude length scale (min = log(2.6/360))
        #elif np.any((-4.28 > p[2]) + (p[2] > 0.)): # longitude length scale (min = log(5/360))
            lnprior = -np.inf
        else:
            lnprior = 0.
        return lnprior

    def lnlike_gp(self, p, sigma):
        """ Calcuates prior and likelihood of GP
        :param p: GP hyperparameters
        :param sigma: noise sigma
        """
        #Uncertainty of y:
        err_blr = sigma
        # Update the kernel and compute the lnlikelihood:
        if george.__version__ < '0.3.0':
            self.kernel.pars = np.exp(p)
        else:
            self.kernel.set_parameter_vector(p)
        #gp = george.GP(self.kernel, mean=np.mean(self.residual_blr))
        gp = george.GP(self.kernel, mean=0.)
        try:
            gp.compute(self.X_gp, err_blr)
            # Note that this can be speed up with pre-calculating covariance matrix and then updating 
            # only uncertainties by adding in quadrature to the diagonal of the covariance matrix.
            # To Do: replace george with manually implemented GP using cholesky decomposition
            if george.__version__ < '0.3.0':
                gplnlike = gp.lnlikelihood(self.residual_blr, quiet=True)
            else:
                gplnlike = gp.log_likelihood(self.residual_blr, quiet=True)
        except:
            print("Problem computing GP likelihood")
            gplnlike = -np.inf
        return gplnlike


    def calc_covariance_matrix(self, data, p, sigma):
        pos = data.copy()
        mag = np.exp(p[0])
        scx = np.exp(p[1])
        scy = np.exp(p[2]) 
        pos[:,0] = pos[:,0] / scx
        pos[:,1] = pos[:,1] / scy
        #pos[:,1] = pos[:,1] / scx # only one length scale
        dist_matrix = pairwise_distances(pos)
        cov_SqrdExp = mag * np.exp(-0.5*dist_matrix**2)
        #print("Squared Exponential")
        #print(cov_SqrdExp)
        ndata = pos.shape[0]
        cov_WhiteNoise = sigma**2 * np.identity(ndata)
        return cov_WhiteNoise + cov_SqrdExp
        #return cov_WhiteNoise # for BLR only


    def lnlikelihood(self, resid, alpha, beta, sigma, p):
        """ Calculates likelihood
        :param resid: residuals from data given proposed BLR
        :param alpha: constant
        :param beta: coefficient parameters
        :param sigma: noise sigma
        :param p: GP hyperparameters
        """
        #y_model = alpha + np.sum(beta * self.X_blr, axis=1)
        #residuals = (self.y - y_model)
        residuals = resid
        covariance_matrix = self.calc_covariance_matrix(self.X_gp, p, sigma)
        #lower = np.linalg.cholesky(covariance_matrix)
        #if(np.any(np.diag(lower)==0.)):
        #    print("Found zeros in the diagonal elements of the lower-triangular cholesky decomposition of the covariane matrix")
        try:
            (sign,logdetcov) = np.linalg.slogdet(covariance_matrix) #sign should be positive
            CinvRes = np.linalg.solve(covariance_matrix,residuals)
            lnlike =  -0.5 * (logdetcov + residuals.T.dot(CinvRes) + 2*np.log(2*np.pi))
        except:
            print("Problem computing likelihood")
            print("Params (alpha, sigma, beta vec, GP vec):")
            print(np.hstack((alpha, sigma, beta, p)))
            lnlike = -np.inf
        return lnlike


    def lnprior_sigma(self, sigma):
        """
        Prior for gaussian noise
        :param sigma: noise sigma
        """
        if np.any((1e-3 > sigma) + (sigma > self.y_std)):
            ln_prob = - np.inf
        else:
            type='uniform'
            if(type=='uniform'):
                ln_prob = 0.
            elif(type=='separable'):
                ln_prob = ln_prob - np.log(sigma) # independent sigma prior
        return ln_prob


    def lnprior_blr(self, alpha, beta, sigma):
        """
        Prior for Bayesian Linear Regression 
        :param beta: coefficient parameters
        :param sigma: noise sigma
        """
        # log of prior for BLR; see paper.
        #if sigma <= 0.:
        if np.any((1e-3 > sigma) + (sigma > self.y_std)):
            ln_prob = - np.inf
        #elif np.any((-5 > alpha) + (alpha > 5)):
        #    ln_prob = - np.inf
        elif np.any((-10 > beta) + (beta > 10)):
            ln_prob = -np.inf
        else:
            type='separable'
            if(type=='uniform'):
                ln_prob = 0.
            elif(type=='separable'):
                lnp_sigma = - np.log(sigma) # independent sigma prior: Jeffrey's on sigma
                #lnp_sigma = 0.2*np.log(sigma) - sigma # independent sigma prior
                lnp_betas = -0.5 * (beta.T.dot(beta)) / np.var(self.y)
                ln_prob = lnp_sigma + lnp_betas
            elif(type=='original'):
                q = self.ndim_blr
                #q = len(beta)  # length of all features
                c = self.X_blr.shape[0] # length of data points
                XTX = self.X_blr.T.dot(self.X_blr)
                XTy = self.X_blr.T.dot(self.y)
                m = np.linalg.solve(XTX,XTy)
                #m = np.linalg.pinv(XTX).dot(XTy) # array with len of q
                fact1 = (beta - m) # array with len of q
                fact2 = 1./c * self.X_blr.T.dot(self.X_blr).dot(fact1) #  array with len of q
                ln_prob = -(q/2+1) * np.log(c/(c+1.)*sigma**2) - 0.5 * fact1.T.dot(fact2)/sigma**2
        return ln_prob


    def lnlikelihood_blr(self, alpha, beta, sigma):
        """ Returns Log Likelihood of BLR and residual (for updating GP)
        :param alpha: constant parameter
        :param beta: coefficient parameters 
        :param sigma: noise sigma
        """
        y_model = alpha + np.sum(beta * self.X_blr, axis=1)
        resid = (self.y - y_model)
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + resid ** 2 / sigma ** 2), resid


    def lnprob_blr(self, alpha, beta, sigma):
        # Returns sum of BLR Likelihood and Prior as well as residual
        lnlik_blr, resid = self.lnlikelihood_blr(alpha, beta, sigma)
        return self.lnprior_blr(alpha, beta, sigma) + lnlik_blr, resid


    def lnprob(self, params):
        # Returns the combined log posterior of BLR and GP
        alpha = params[0] # BLR constant
        sigma = params[1] # Sigma
        beta = params[2:self.ndim_blr+2]
        p = params[self.ndim_blr+2:]
        lnpost_blr, resid_blr = self.lnprob_blr(alpha, beta, sigma)
        #self.residual_blr = resid_blr
        #return lnpost_blr + self.lnprior_gp(p) # just keep GP hyperparameters under control
        #return lnpost_blr + self.lnprior_gp(p) + self.lnlike_gp(p, sigma) 
        #return self.lnprior_blr(alpha, beta, sigma) + self.lnprior_gp(p) + self.lnlike_gp(p, sigma)
        return self.lnprior_blr(alpha, beta, sigma) + self.lnprior_gp(p) + self.lnlikelihood(resid_blr, alpha, beta, sigma, p) 


    def scale_data(self, data):
        # Scales input data with RobustScaler
        robust_scaler = RobustScaler()
        return robust_scaler.fit_transform(data)

    def norm_data(self, data):
        # Normalization of data with MinMaxScaler
        min_max_scaler = MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(data)
        return data_scaled, min_max_scaler.data_min_, min_max_scaler.data_range_

    def calc_mcmc(self, nwalkers, niter, nburn, split_traintest):
        """ Running the MCMC
        :param nwalkers: Number of walkers, recommended at least 100
        :param niter: Number of iterations, recommend at least 500
        :param nburn: Number of iterations for burn-in phase, recommend at least 100
        :return: None; updates and saves class variables
        """
        print("Data Shape")
        print(self.X_gp.shape)
        self.kernel = self.kernel_gp(kernelname = kernelname)
        self.ndim_gp = len(self.kernel) # same as gpdim
        self.residual_blr = np.zeros_like(self.y)
        # Set up the sampler:
        self.nwalkers = nwalkers
        self.niter = niter
        self.ndim = np.int(self.ndim_gp + self.ndim_blr + 2)
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)
        """
        # Initialize the walkers:
        if george.__version__ < '0.3.0':
            p0_gp = np.log(self.kernel.pars)
        else: 
            p0_gp = self.kernel.get_parameter_vector()
        """
        alpha0 = np.mean(self.y)
        beta0 = np.zeros((self.ndim_blr))
        sigma0 = self.y_std/2.
        p0_gp = -1.*np.ones((self.ndim_gp))
        p0_gp[0] = 2.*np.log(sigma0/4.)
        p0_comb = np.hstack((alpha0, sigma0, beta0, p0_gp))
        print("Initial proposal (alpha, sigma, beta vec, GP vec):")
        print(p0_comb)
        print("Initial log posterior function call")
        print(self.lnprob(p0_comb))
        p0 = [p0_comb + 1e-4 * np.random.randn(self.ndim) for i in range(self.nwalkers)]

        print("Estimating MCMC time...")
        start_time = dt.datetime.now()
        _, _, _ = sampler.run_mcmc(p0, 10)
        # Reset the chain to remove the burn-in samples.
        sampler.reset()
        burn_time = (dt.datetime.now() - start_time).seconds
        print('Estimated time till completed: {} seconds '.format(burn_time * (self.niter+nburn) / 10.))

        print("Running burn-in...")
        #print("Running burn-in 1 of 2...")
        #p0, lp, state = sampler.run_mcmc(p0, int(nburn/2))
        p0, lp, state = sampler.run_mcmc(p0, nburn)
        p0_best = p0[np.argmax(lp)]
        print("Post-burn-in re-proposal (alpha, sigma, beta vec, GP vec):")
        print(p0_best)
        print("Post-burn-in log posterior function call")
        print(self.lnprob(p0_best))
        sampler.reset()         # Reset the chain to remove the burn-in samples.
        #print("Running burn-in 2 of 2...")
        #p0 = p0_best + 1e-8*np.random.randn(self.nwalkers, self.ndim)
        #p0, lp, state = sampler.run_mcmc(p0, int(nburn/2))
        #sampler.reset()         # Reset the chain to remove the burn-in samples.


        print("Running MCMC ...")
        pos, prob, state = sampler.run_mcmc(p0, self.niter, rstate0=state)
        # Save the mean acceptance fraction:
        af = sampler.acceptance_fraction
        self.accept_fr = af
        # Get the best model parameters and their respective errors:
        self.sampler_chain = sampler.chain
        self.sampler_flatchain = sampler.flatchain
        maxprob_index = np.argmax(prob)
        #self.pos_fit = pos
        #self.prob_fit = prob
        # save parameters with largest probability:
        self.params_fit = pos[maxprob_index]
        self.params_mean = np.mean(pos, axis=0)
        # save percentile of posterior distribution:
        self.params_50per = [np.percentile(sampler.flatchain[:, i], 50) for i in range(self.ndim)]
        self.params_2per = [np.percentile(sampler.flatchain[:, i], 2) for i in range(self.ndim)]
        self.params_16per = [np.percentile(sampler.flatchain[:, i], 16) for i in range(self.ndim)]
        self.params_84per = [np.percentile(sampler.flatchain[:, i], 84) for i in range(self.ndim)]
        self.params_97per = [np.percentile(sampler.flatchain[:, i], 98) for i in range(self.ndim)]
        # save standard deviation:
        self.errors_fit = np.asarray([sampler.flatchain[:, i].std() for i in range(self.ndim)])
        # save parameters:
        self.alpha_fit, self.alpha_err = self.params_50per[0], self.errors_fit[0]
        self.sigma_fit, self.sigma_err = self.params_50per[1], self.errors_fit[1]
        self.beta_fit, self.beta_err = self.params_50per[2:self.ndim_blr + 2], self.errors_fit[2:self.ndim_blr + 2]
        self.p_fit, self.p_err = self.params_50per[self.ndim_blr + 2:], self.errors_fit[self.ndim_blr + 2:]
        self.beta_fit = self.beta_fit 

        print("Median aposteriori params (alpha, sigma, beta vec, GP vec):")
        print(self.params_50per)
        print("Median aposteriori log posterior function call")
        print(self.lnprob(np.hstack((self.params_50per))))

        # Calculate models and residuals for train data
        self.mu_blr = self.predict_blr(self.X_blr, self.alpha_fit, self.beta_fit) # BLR Model 
        self.residual_blr = (self.y - self.mu_blr)
        """
        #gp = george.GP(self.kernel, mean=np.mean(self.residual_blr))
        gp = george.GP(self.kernel, mean=0.)
        if george.__version__ < '0.3.0':
            gp.kernel.pars = np.exp(self.p_fit)
        else:
            gp.kernel.set_parameter_vector(self.p_fit)
        gp.compute(self.X_gp, self.sigma_fit)
        self.mu_gp, cov_gp = gp.predict(self.residual_blr, self.X_gp) # GP Model
        #self.mu_gp, cov_gp = gp.predict(self.y, self.X_gp) # GP Model
        """
        cov_gp = self.calc_covariance_matrix(self.X_gp, self.p_fit, self.sigma_fit)
        Lower = np.linalg.cholesky(cov_gp)
        self.y_model = self.mu_blr + Lower.T.dot(np.random.normal(0,1,self.ndata))
        self.std_gp = np.sqrt(np.diag(cov_gp)) # standard deviation of GP
        #self.y_model = np.random.multivariate_normal(self.mu_blr, cov_gp)
        #self.y_model = self.mu_gp + self.mu_blr # Final Model 
        #self.y_model = self.mu_blr # Final Model 
        #self.y_model = self.mu_gp

        # Print some MCMC results and additionally saved in text file:
        print("---- MCMC Results and Parameters ----")
        self.results_file.write('---- MCMC Results and Parameters ----\n')
        print("Mean acceptance fraction:", np.mean(af))
        self.results_file.write('Mean acceptance fraction: {0} \n'.format(np.mean(af)))
        """
        print("Kernel: ", gp.kernel)
        self.results_file.write('Kernel: {0} \n'.format(gp.kernel))
        """
        print("alpha, err:", round(self.alpha_fit,3), round(self.alpha_err,3))
        self.results_file.write('alpha: {0} , err: {1} \n'.format(round(self.alpha_fit,3), round(self.alpha_err,3)))
        for i in range(len(self.beta_fit)):
            print('beta'+str(i)+' , err:', round(self.beta_fit[i],3), round(self.beta_err[i],3))
            self.results_file.write('beta {0}: {1} , err: {2} \n'.format(str(i), round(self.beta_fit[i],3), round(self.beta_err[i],3)))
        print('sigma, err:', round(self.sigma_fit,3), round(self.sigma_err,3))
        self.results_file.write('sigma: {0} , err: {1} \n'.format(round(self.sigma_fit,3), round(self.sigma_err,3)))
        for i in range(len(self.p_fit)):
            print('GP'+str(i)+' , err:', round(self.p_fit[i],3), round(self.p_err[i],3))
            self.results_file.write('GP {0}: {1} , err: {2} \n'.format(str(i), round(self.p_fit[i],3), round(self.p_err[i],3)))
        #print("Model lnlikelihood: ", prob[maxprob_index])
        #self.results_file.write('Model lnlikelihood: {0} \n'.format(prob[maxprob_index]))
        print("Std GP: ", np.mean(self.std_gp))
        self.results_file.write('Std GP: {0} \n'.format(np.mean(self.std_gp)))
        """
        if george.__version__ < '0.3.0':
            print("GP lnlikelihood:", gp.lnlikelihood(self.y))
            self.results_file.write('GP lnlikelihood: {0} \n'.format(gp.lnlikelihood(self.y)))
        else:
            print("GP lnlikelihood:", gp.log_likelihood(self.y))
            self.results_file.write('GP lnlikelihood: {0} \n'.format(gp.log_likelihood(self.y)))
        """
        # Calculate models and residuals for test data
        if split_traintest > 0.:
            self.mu_blr_test = self.predict_blr(self.X_blr_test, self.alpha_fit, self.beta_fit)  # BLR Model
            self.residual_blr_test = (self.y_test - self.mu_blr_test)
            """
            #gp = george.GP(self.kernel, mean=np.mean(self.residual_blr_test))
            gp = george.GP(self.kernel, mean=0.)
            gp.compute(self.X_gp, self.sigma_fit)
            #self.mu_gp_test, _ = gp.predict(self.residual_blr_test, self.X_gp_test)  # GP Model 
            self.mu_gp_test, _ = gp.predict(self.residual_blr, self.X_gp_test)  # GP Model 
            """
            cov_gp_test = self.calc_covariance_matrix(self.X_gp_test, self.p_fit, self.sigma_fit)
            Lower = np.linalg.cholesky(cov_gp_test)
            self.y_model_test = self.mu_blr_test + Lower.T.dot(np.random.normal(0,1,len(self.y_test)))
            #self.y_model_test = np.random.multivariate_normal(self.mu_blr_test, cov_gp_test)
            #self.y_model_test = self.mu_gp_test + self.mu_blr_test  # Final Model 
            #self.y_model_test = self.mu_blr_test  # Final Model 
            #self.y_model_test = self.mu_gp_test  # Final Model 
        else:
            self.mu_blr_test, self.mu_gp_test, self.residual_blr_test, self.y_model_test = np.zeros(4)


    def calc_residual(self, plot3d=False):
        """ Print and plots residual
        :param plot3d: boolean (True/False) whether residual of GP is plotted in 3D, default=False
        :return: None
        """
        self.residual = self.y - self.y_model
        self.residual_test = self.y_test - self.y_model_test
        print('Mean abs Residual train: ', round(np.mean(abs(self.residual)),3))
        print('Mean abs Residual BLR train: ', round(np.mean(abs(self.residual_blr)),3))
        #print('Mean abs Residual GP train: ',  round(np.mean(abs(self.y - self.mu_gp)),3))
        self.results_file.write('Mean abs Residual train: {0} \n'.format(round(np.mean(abs(self.residual)),3)))
        self.results_file.write('Mean abs Residual BLR train: {0} \n'.format(round(np.mean(abs(self.residual_blr)),3)))
        #self.results_file.write('Mean abs Residual GP train: {0} \n'.format(round(np.mean(abs(self.y - self.mu_gp)),3)))
        rmse_train = np.sqrt(np.sum(self.residual**2)) / len(self.residual)
        rmse_test = np.sqrt(np.sum(self.residual_test**2)) / len(self.residual_test)
        print('RMSE train: ', round(rmse_train, 3))
        print('RMSE test: ', round(rmse_test, 3))
        self.results_file.write('RMSE train: {0} \n'.format(round(rmse_train,3)))
        self.results_file.write('RMSE test: {0} \n'.format(round(rmse_test,3)))
        #print('RMSE: ', round(np.sqrt(np.sum(self.residual**2) / len(self.residual)),3))
        #self.results_file.write('Mean abs Residual: {0} \n'.format(round(np.sqrt(np.sum(self.residual**2) / len(self.residual)),3)))
        # self.results_file.close()
        #Optional: make residual map of spatial GP component:
        if plot3d:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plt.hold(True)
            #ax.plot_surface(xfield, yfield, model, cmap=cm.hot)
            ax.scatter(self.X_gp[:,0], self.X_gp[:,1], self.residual, c='r', marker='o')
            ax.set_xlabel('X Norm')
            ax.set_ylabel('Y Norm')
            ax.set_zlabel('Residual '+ target_desc)

    def create_param_csv(self, par_list, icross = 0):
        """ stores model parameter stats and sampler chain
        :param par_list: list of strings for parameters
        :param icross: current number of cross-validation
        :return: None
        """
        par_array = np.vstack([par_list, self.params_fit, self.params_mean, self.errors_fit,
                  self.params_50per, self.params_2per, self.params_16per, self.params_84per, self.params_97per]).T
        header = "par_name, max_prob par, walker mean par, std, par_50per, par_2per, par_16per, par_84per, par_97per"
        if nfold_cross > 1: 
            np.savetxt(self.outmcmc+'result_parameters_' + str(icross) + '.csv', par_array, delimiter=",", header=header, fmt='%s')
            np.save(self.outmcmc + 'sampler_chain_' + str(icross), self.sampler_chain) # saves as npy file
            np.save(self.outmcmc + 'ymodel_chain_area_' + str(icross), self.mu_i)  # saves as npy file
        else:
            np.savetxt(self.outmcmc+'result_parameters.csv', par_array, delimiter=",", header=header, fmt='%s')
            np.save(self.outmcmc + 'sampler_chain', self.sampler_chain) # saves as npy file
            np.save(self.outmcmc + 'ymodel_chain_area', self.mu_i)  # saves as npy file


    def create_scaling_csv(self, par_list, min_list, range_list, par_type='par', icross = 0):
        """ stores model parameter stats and sampler chain
        :param par_list: list of strings for parameters
        :param icross: current number of cross-validation
        :return: None
        """
        par_array = np.vstack([par_list, min_list, range_list]).T
        header = "par_name,minimum,range"
        if nfold_cross > 1: 
            np.savetxt(self.outmcmc+'minmaxscale_' + par_type + '_'+str(icross)+'.csv', par_array, delimiter=",", header=header, fmt='%s')
        else:
            np.savetxt(self.outmcmc+'minmaxscale_' + par_type + '.csv', par_array, delimiter=",", header=header, fmt='%s')


    def vis_mc(self):
        # Basic plot data and model, works so far only with  testdata
        #self.plot_2D(self.X_gp, self.model_gp, self.data_gp)
        # calculate residual map
        res_map = (self.data - self.model) / self.data
        plt.figure(3)
        plt.imshow(res_map)
        plt.colorbar()
        plt.title('Residual/Data Map')
        #plt.show()

    def calc_samples(self, nitersample=100):
        """ Calculates the distribution of model y, residual, and percentiles over multiple samples
        over all walkers times numbers of iterations to include at end of chain.

        :param nitersample: number of iterations at chain end to be included in sample, default 100;
        Set to 1 if only samples over walkers required, will take only last entries in chain.
        :return: None; saves values in class variable mu_i, residual_i, and percentiles
        """
        if nitersample > self.niter:
            print("Chosen number of iterations for samples larger than chain. Setting to maximal number of iterations")
            nitersample = self.niter
        if nitersample <= 0:
            print("Chosen number of iterations for samples too small. Setting to minimum size of 1 chain iteration.")
            nitersample = 1
        nsamples = self.nwalkers * nitersample
        print("Number of chosen samples for evaluation: ,", nsamples, " Calculating y model values ...")
        nlim = self.niter - nitersample
        choose_samples = np.random.choice(self.niter, nitersample, replace=False)
        self.samples_i = self.sampler_chain[:, choose_samples, :].reshape((-1, self.ndim))
        #self.samples_i = self.sampler_chain[:, nlim:, :].reshape((-1, self.ndim))
        self.mu_i = np.zeros((nsamples, self.ndata))
        self.residual_i = np.zeros_like(self.mu_i)
        self.mu_i_test  = np.zeros((nsamples, len(self.y_test)))
        for i in range(self.samples_i.shape[0]):
            alpha = self.samples_i[i,0]
            sigma = self.samples_i[i,1]
            beta = self.samples_i[i, 2:self.ndim_blr+2]
            gphp = self.samples_i[i, self.ndim_blr+2:]
            # Calculate for train data set:
            y_blr = self.predict_blr(self.X_blr, alpha, beta)
            resid_blr = self.y - y_blr
            """
            #gp = george.GP(self.kernel, mean=np.mean(resid_blr))
            gp = george.GP(self.kernel, mean=0.)
            if george.__version__ < '0.3.0':
                gp.kernel.pars = np.exp(self.samples_i[i, self.ndim_blr + 2:])
            else:
                gp.kernel.set_parameter_vector(self.samples_i[i, self.ndim_blr + 2:])
            try:
                gp.compute(self.X_gp, sigma)
                mu_gp, _ = gp.predict(resid_blr, self.X_gp)
                self.mu_i[i, :] = y_blr + mu_gp
                #mu_gp, _ = gp.predict(self.y, self.X_gp)
                #self.mu_i[i, :] = mu_gp
            except:
                self.mu_i[i,:] = y_blr
            """
            cov_gp = self.calc_covariance_matrix(self.X_gp, gphp, sigma)
            Lower = np.linalg.cholesky(cov_gp)
            self.mu_i[i, :] = y_blr + Lower.T.dot(np.random.normal(0,1,self.ndata))
            #self.mu_i[i,:] = np.random.multivariate_normal(y_blr, cov_gp)
            self.residual_i[i] = np.sum(self.y - self.mu_i[i, :])
            # Calculate for test data:
            y_blr_test = self.predict_blr(self.X_blr_test, alpha, beta)
            resid_blr_test = self.y_test - y_blr_test
            """
            #gp = george.GP(self.kernel, mean=np.mean(resid_blr_test))
            gp = george.GP(self.kernel, mean=0.)
            try:
                gp.compute(self.X_gp, sigma)
                #mu_gp_test, _ = gp.predict(resid_blr_test, self.X_gp_test)
                mu_gp_test, _ = gp.predict(resid_blr, self.X_gp_test)
                self.mu_i_test[i, :] = y_blr_test + mu_gp_test
                #mu_gp_test, _ = gp.predict(self.y_test, self.X_gp_test)
                #self.mu_i_test[i, :] = mu_gp_test
            except:
                self.mu_i_test[i, :] = y_blr_test
            """
            cov_gp_test = self.calc_covariance_matrix(self.X_gp_test, gphp, sigma)
            Lower = np.linalg.cholesky(cov_gp_test)
            self.mu_i_test[i, :] = y_blr_test + Lower.T.dot(np.random.normal(0,1,len(self.y_test)))
            #self.mu_i_test[i, :] = np.random.multivariate_normal(y_blr_test, cov_gp_test)
        self.perc2_area = np.asarray([np.percentile(self.mu_i[:, i], 2) for i in range(self.ndata)]) # rounded from 2.3
        self.perc16_area = np.asarray([np.percentile(self.mu_i[:, i], 16) for i in range(self.ndata)])
        self.perc50_area = np.asarray([np.percentile(self.mu_i[:, i], 50) for i in range(self.ndata)])
        self.perc84_area = np.asarray([np.percentile(self.mu_i[:, i], 84) for i in range(self.ndata)])
        self.perc97_area = np.asarray([np.percentile(self.mu_i[:, i], 98) for i in range(self.ndata)]) # rounded from 97.7
        #print("Number of train areas within 2 and 97 percentile:", len(self.y[(self.y > self.perc2_area) & (self.y <= self.perc97_area)]))
        #print("Number of train areas within 16 and 84 percentile:", len(self.y[(self.y > self.perc16_area) & (self.y <= self.perc84_area)]))
        self.ymodel_mean = np.mean(self.mu_i, axis=0)
        self.ymodel_std = np.std(self.mu_i, axis=0)
        ci95_train = len(self.y[(self.y >= (self.ymodel_mean - 2 * self.ymodel_std)) & (self.y < (self.ymodel_mean + 2 * self.ymodel_std))]) \
                     * 1. / self.ndata
        self.ymodel_mean_test = np.mean(self.mu_i_test, axis=0)
        self.ymodel_std_test = np.std(self.mu_i_test, axis=0)
        ci95_test = len(self.y_test[(self.y_test >= (self.ymodel_mean_test - 2 * self.ymodel_std_test)) & (self.y_test < (self.ymodel_mean_test + 2 * self.ymodel_std_test))]) \
                    * 1. / len(self.y_test)
        print('Percent of train locations in 95 percent CI: ', ci95_train * 100.)
        self.results_file.write('Percent of train locations in 95 percent CI: {0} \n'.format(ci95_train * 100.))
        print('Percent of test locations in 95 percent CI: ', ci95_test * 100.)
        self.results_file.write('Percent of test locations in 95 percent CI: {0} \n'.format(ci95_test * 100.))  
        per97_train = np.percentile(self.mu_i, 97, axis=0)
        per2_train = np.percentile(self.mu_i, 2, axis=0)
        per97_test = np.percentile(self.mu_i_test, 97, axis=0)
        per2_test = np.percentile(self.mu_i_test, 2, axis=0)
       # ci_per95_train = len(self.y[(self.y > per2_train) & (self.y < per97_train)]) * 1. / self.ndata
       # ci_per95_test = len(self.y_test[(self.y_test > per2_test) & (self.y_test < per97_test)])  * 1./ len(self.y_test)
       # print('Percent of train locations in 95 percent CI: ', np.round(ci_per95_train * 100.,3))
       # self.results_file.write('Percent of train locations in 95 percent CI: {0} \n'.format(np.round(ci_per95_train * 100.,3)))
       # print('Percent of test locations in 95 percent CI: ', np.round(ci_per95_test * 100.,3))
       # self.results_file.write('Percent of test locations in 95 percent CI: {0} \n'.format(np.round(ci_per95_test * 100.,3)))  


    def plot_hist(self, par_list):
        """ Plots histograms for samples of parameters, model results, residuals, and test data

        :param par_list: list of string for parameter names
        :return: None; creates and saves plots in main class output directory
        """
        print("Plotting histograms and graphs...")
        outdir = self.outmcmc+'histograms/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # plotting histograms for parameters
        for i in range(self.ndim):
            plt.figure(i)
            dist = self.samples_i[:, i]
            plt.hist(dist, 30, facecolor='blue', alpha=0.5)
            if i ==0: plt.title('Alpha')
            if i ==1: plt.title('Sigma')
            if (i > 1) and (i<self.ndim_blr + 2): plt.title(par_list[i])
            if i >= self.ndim_blr + 2:
                plt.title('GP par: '+str(i-int(self.ndim_blr) - 1))
            plt.axvline(self.params_fit[i], c='r', ls='-')
            plt.axvline(self.params_mean[i], c='b', ls='-')
            plt.axvline(self.params_mean[i] + self.errors_fit[i], c='b', ls='--')
            plt.axvline(self.params_mean[i] - self.errors_fit[i], c='b', ls='--')
            plt.axvline(0., c='k', ls='--')
            plt.draw()
            plt.savefig(outdir+'hist_'+par_list[i]+'.png')
        # plot observed vs modeled
        xmin = np.min(np.concatenate([self.y,self.y_test]))
        xmax = np.max(np.concatenate([self.y,self.y_test]))
        xdiff = xmax - xmin
        ymin = np.min(np.concatenate([self.y_model,self.y_model_test]))
        ymax = np.max(np.concatenate([self.y_model,self.y_model_test]))
        ydiff = ymax - ymin
        x_range = [xmin - xdiff*0.2, xmax + xdiff*0.2]
        y_range = [ymin - ydiff*0.2, ymax + ydiff*0.2]
        plt.clf()
        plt.plot(x_range, y_range, '.', alpha=0.0)
        #plt.plot(self.y, self.y_model, 'o', c='b',label='Train', ms=3, alpha=0.5)
        plt.plot(self.y, self.ymodel_mean, 'o', c='b',label='Train', ms=3, alpha=0.5)
        #plt.plot(self.y_test, self.y_model_test, 'o', c='r',label='Test', ms=3, alpha=0.5)
        plt.plot(self.y_test, self.ymodel_mean_test, 'o', c='r',label='Test', ms=3, alpha=0.5)
        plt.plot(self.y, self.y, 'k--', label='Perfect Prediction')
        plt.xlabel('Observed y')
        plt.ylabel('Predicted y')
        plt.title('Predicted vs Ground Truth')
        plt.legend(loc='upper left', numpoints=1)
        ax = plt.axes([0.63, 0.15, 0.25, 0.25], frameon=True)
        ax.hist(self.residual, alpha=0.5, bins=16, color='b', stacked=True, normed=True)
        ax.hist(self.residual_test, alpha=0.5, bins=16, color='r', stacked=True, normed=True)
        #ax.xaxis.set_ticks(np.arange(-2, 2, 1))
        ax.set_title('Residual')
        plt.draw()
        plt.savefig(self.outmcmc + 'Data_vs_Model_1d.png')
        # Plot Prediction & Uncertainty for Test locations
        xmin = np.min(self.y_test)
        xmax = np.max(self.y_test)
        xdiff = xmax - xmin
        ymin = np.min(self.y_model_test)
        ymax = np.max(self.y_model_test)
        ydiff = ymax - ymin
        x_range = [xmin - xdiff*0.2, xmax + xdiff*0.2]
        y_range = [ymin - ydiff*0.2, ymax + ydiff*0.2]
        plt.clf()
        plt.plot(x_range, y_range, '.', alpha=0.0)
        #plt.plot(self.y_test, self.y_model_test, 'o', c='k',label='Test', ms=4, alpha=0.5)
        plt.plot(self.y_test, self.ymodel_mean_test, 'o', c='k',label='Prediction', ms=4, alpha=0.5)
        plt.plot(self.y_test, self.ymodel_mean_test+2.*self.ymodel_std_test, 'o', c='b',label='Pred + 2std', ms=3, alpha=0.5)
        plt.plot(self.y_test, self.ymodel_mean_test-2.*self.ymodel_std_test, 'o', c='r',label='Pred - 2std', ms=3, alpha=0.5)
        plt.plot(self.y, self.y, 'k--', label='Perfect Prediction')
        plt.xlabel('Observed y')
        plt.ylabel('Predicted y')
        plt.title('Predicted vs Ground Truth (Test)')
        plt.legend(loc='upper left', numpoints=1)
        plt.draw()
        plt.savefig(self.outmcmc + 'Data_vs_Model_Test.png')
        # plot more histograms:
        # plt.clf()
        # plt.hist(self.residual_i, 30, facecolor='blue', alpha=0.5)
        # plt.title('Sum Residual y-model')
        # plt.ylabel('N Samples')
        # plt.draw()
        # plt.savefig(outdir + 'hist_residual.png')
        plt.clf()
        plt.plot(self.y, self.mu_blr, 'o', ms=3)
        plt.plot(self.y, self.y, 'k--')
        plt.xlabel('Actual '+ target_desc)
        plt.ylabel('Predicted '+ target_desc)
        plt.title('Predicted from Demographics vs Ground Truth')
        plt.draw()
        plt.savefig(self.outmcmc + 'Data_ModelBLR_1d.png')
        plt.clf()
        plt.plot(self.y_model, self.residual, 'o', ms=3)
        plt.plot([min(self.y_model),max(self.y_model)],[0,0],'k--')
        plt.xlabel('Predicted '+ target_desc)
        plt.ylabel('Residual')
        plt.draw()
        plt.savefig(self.outmcmc + 'Model_Residual_1d.png')
        plt.clf()
        plt.plot(self.y_model_test, self.residual_test, 'o', ms=3)
        plt.plot([min(self.y_model_test),max(self.y_model_test)],[0,0],'k--')
        plt.xlabel('Test Predicted '+ target_desc)
        plt.ylabel('Test Residual')
        plt.draw()
        plt.savefig(self.outmcmc + 'Model_Residual_Test_1d.png')


    def plot_diagr(self, namelist):
        """ Plots coefficient of linear regression for features
        :param namelist: list of parameter names in string format
        :return: None
        """
        print('Plotting sorted correlations in diagram...')
        ticks = np.asarray(namelist)
        data = np.asarray([self.params_50per, self.params_2per, self.params_16per, self.params_84per, self.params_97per])
        box = data[:,2:(self.ndim_blr + 2)]
        sort = np.argsort(np.mean(box,axis=0))
        box_s = box[:,sort]
        plt.figure()
        plt.clf()
        bp = plt.boxplot(box_s, vert=False)
        set_box_color(bp, 'blue')
        plt.yticks(np.linspace(1,len(ticks), len(ticks)), ticks[sort])
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.axvline(0, color='k', ls='--')
        #plt.xlabel('Feature X-y Correlation')
        plt.xlabel('Regression Coefficient')
       # plt.tight_layout()
        plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(self.outmcmc + 'feature_y_correlation.png')


    def plot_corner(self, par_list):
        """ Creates corner plot of MCMC results for parameters
        :param par_list: list of string for parameter names
        :return: no return, saves plot in main output directory
        """
        print("Plotting corner diagrams, be patient ....")
        plt.clf()
        import corner
        outdir = self.outmcmc
        samples = self.sampler_chain[:, :, :].reshape((-1, self.ndim))
        fig = corner.corner(samples, labels=par_list, truths=self.params_50per)
        fig.savefig(outdir + "cornerplot_all.png")

    def plot_niter(self, par_list):
        """ plot parameters as function of MCMC iteration including all walkers
        :param par_list: list of string for parameter names
        :return: no return variable, saves plots in main output directory
        """
        print("Plotting parameters along niter ...")
        outdir = self.outmcmc + 'plots_niter/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        niter = np.linspace(0, self.niter - 1, self.niter)
        for i in range(self.ndim):
            plt.clf()
            for j in range(self.nwalkers):
                plt.plot(niter, self.sampler_chain[j, :, i], c='k')
            plt.axhline(self.params_50per[i], c='b', ls='-')
            plt.axhline(self.params_16per[i], c='b', ls='--')
            plt.axhline(self.params_84per[i], c='b', ls='--')
            plt.xlabel("N Iteration")
            plt.ylabel(par_list[i])
            plt.savefig(outdir + 'niter_' + par_list[i] + '.png')
