'''
KISSGP+NUTS.py

Script for learning KISS-GP with a real data example.
Uses GPytorch's KISS GP method to solve for the hyperparameters
Uses NUTS as the HMC algorithm.

Created by: Benjamin Metha
Last updated: Mar 05, 2025
'''
# Import relevant packages
# import math
import torch
import gpytorch
#from matplotlib import pyplot as plt
import numpy as np
import pyro
from pyro.infer.mcmc import NUTS, MCMC
import GeoGals as gg
import pandas as pd
import argparse

########################
##	 Read arguments	  ##
########################

parser = argparse.ArgumentParser()

parser.add_argument("-g",
			"--gal-name",
			help="What is the name of the galaxy you are interested in? Format as Nxxxx.",
			required=True,
			type=str,
			dest='gal_name',
			nargs=1)
			
parser.add_argument("-d",
			"--diagnostic",
			help="What SEL diagnostic should be used to estimate the metallicity?",
			required=True,
			type=str,
			dest='diag',
			nargs=1)
			
parser.add_argument("-l",
			"--chain-length",
			help="How long will we run these chains for?",
			required=True,
			type=int,
			dest='length',
			nargs=1)
			
args = parser.parse_args()
			
for i in args.gal_name:
	gal_ID = i

for i in args.length:
	N_SAMPLES = i
	
for i in args.diag:
	diag = i

output_path = 'Results'

#######################
## Wrangle real data ##
#######################

Hii_df = gg.open_Hii_df(gal_ID)
meta   = gg.meta_getter(gal_ID)

r_char = 0.4*meta['R25_kpc']

wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])

# Get relevant vectors for Gpytorch
XY	= gg.RA_DEC_to_XY(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
Z	= Hii_df['Z_'+diag][wanted_spaxels]
e_Z = Hii_df['e_Z_'+diag][wanted_spaxels]

# Convert these to torch tensors
XY	  = torch.from_numpy(XY)
Z	  = torch.tensor(Z.values)
var_Z = torch.tensor(e_Z.values**2)

# Make them floats, not doubles (else code breaks):
XY	  = XY.type(torch.FloatTensor)
Z	  = Z.type(torch.FloatTensor)
var_Z = var_Z.type(torch.FloatTensor)

#######################
##					 ##
##	   Set up GP	 ##
##					 ##
#######################

# Create class for mean
class RadialLinearMean(gpytorch.means.Mean):
	def __init__(self):
		super().__init__()
		# Learnable coefficients
		self.gradZ = torch.nn.Parameter(torch.randn(1))	 # Slope
		self.Z_char = torch.nn.Parameter(torch.randn(1))  # Intercept
	
	def forward(self, x):
		# Assuming x has shape [N, D], where D = 2 (at least x and y)
		x_coords = x[:, 0]	# First dimension: x
		y_coords = x[:, 1]	# Second dimension: y
		r = torch.sqrt(x_coords**2 + y_coords**2)  # Compute radial distance
		return self.gradZ * (r - r_char) + self.Z_char

class GPRegressionModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

		# SKI requires a grid size hyperparameter. This util can help with that
		grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

		self.mean_module = RadialLinearMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(
			gpytorch.kernels.GridInterpolationKernel(
				gpytorch.kernels.MaternKernel(nu=0.5), grid_size=grid_size, num_dims=2
			)
		)

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
		
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = var_Z)
model = GPRegressionModel(XY, Z, likelihood)

#######################
##					 ##
##	 Define Priors	 ##
##					 ##
#######################

prior_data = gg.read_ICs(gal_ID, diag)
prior_Z_char = prior_data[0]
prior_gradZ	 = prior_data[1]

delta_Z_char = 0.15
delta_gradZ = 0.05

model.mean_module.register_prior("Z_char_prior", gpytorch.priors.UniformPrior(prior_Z_char - delta_Z_char, prior_Z_char + delta_Z_char), "Z_char")
model.mean_module.register_prior("gradZ_prior",	 gpytorch.priors.UniformPrior(prior_gradZ - delta_gradZ,   prior_gradZ + delta_gradZ), "gradZ")
model.covar_module.base_kernel.base_kernel.register_prior("lengthscale_prior", gpytorch.priors.UniformPrior(0.05, 2), "lengthscale")
# Probably a log-normal prior would be best for outputscale. Try this and see how it goes...
# model.covar_module.register_prior("outputscale_prior", gpytorch.priors.LogNormalPrior(-5, 4), "outputscale")
# Try the alternative:
model.covar_module.register_prior("outputscale_prior", gpytorch.priors.UniformPrior(1e-5, 2), "outputscale")

#######################
##					 ##
##		GO NUTS		 ##
##					 ##
#######################

def pyro_model(x, y):
	with gpytorch.settings.fast_computations(False, False, False):
		with gpytorch.settings.max_cg_iterations(2000):
			sampled_model = model.pyro_sample_from_prior()
			output = sampled_model.likelihood(sampled_model(x))
			pyro.sample("obs", output, obs=y)
	return y

nuts_kernel = NUTS(pyro_model)
mcmc_run = MCMC(nuts_kernel, num_samples=N_SAMPLES, warmup_steps=0, disable_progbar=False)
mcmc_run.run(XY, Z)

########################
##					  ##
##	  Save Results	  ##
##					  ##
########################
result_dict = mcmc_run.get_samples()
pd.to_pickle(result_dict, output_path + '{0}_{1}_results.pkl'.format(gal_ID, diag))
