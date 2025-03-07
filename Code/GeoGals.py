'''
GeoGals

A selection of functions that are helpful for the geostatistical analysis of 
galaxy data.

Created by: Benjamin Metha
Last Updated: Mar 07, 2025
'''

# Note: Not all of these are used... yet.
import numpy as np
from   astropy.io import fits
from   sklearn.metrics.pairwise import euclidean_distances 
import pandas as pd 
from   astropy.wcs import WCS
import astropy.units as u
from   scipy.linalg import cho_factor, cho_solve
from   extinction import ccm89, apply

local_data_path =  '../Data/'

ASEC_PER_RAD = 206265.0

##################
#    Open data   #
##################

# For cleaned data
def open_Hii_df(gal_ID):
	return pd.read_pickle(local_data_path +'Handmade/Hii_dataframes/Z_maps_{0}.pkl'.format(gal_ID))

def open_metadata():
	'''
	Open the metadata file; saves from re-writing this code in every script,
	makes scripts more readable, and future-proofs code if I need to update
	or move the metadata file.
	'''
	meta_df = pd.read_csv(local_data_path+'metadata.csv')
	metadata = meta_df.to_dict(orient='records')
	return metadata
	
def meta_getter(gal_ID):
	metadata = open_metadata()
	meta   = [x for x in metadata if str(gal_ID) in x['Galaxy_ID']][0]
	return meta
	
def read_ICs(gal_ID, diag):
	return np.load(local_data_path + 'gradZ_ICs/{0}_{1}.npy'.format(gal_ID, diag))

#######################
#    Data wrangling   #
#######################

def make_RA_DEC_grid(header):
	'''
	Given a header file, create a grid of RA//DEC for each pixel in that file.
	'''
	world = WCS(header)
	x = np.arange(header['NAXIS1'])
	y = np.arange(header['NAXIS2'])
	X, Y = np.meshgrid(x, y)
	RA_grid, DEC_grid = world.wcs_pix2world(X, Y, 0)
	return RA_grid, DEC_grid	

def RA_DEC_to_radius(RA, DEC, meta):
	return deprojected_distances(RA, DEC, meta['RA'], meta['DEC'], meta).T[0]

def RA_DEC_to_XY(RA, DEC, meta):
	'''
	Takes in list of RA, DEC coordinates and transforms them into a 
	list of deprojected XY values, where X and Y are the distances from the 
	galaxy's centre in units of kpc
	
	Parameters
	----------
	
	RA: ndarray like of shape (N,)
		List of RA values
		
	DEC: ndarray like of shape (N,)
		List of DEC values
		
	meta: dict
		Must contain RA, DEC of the galaxy centre, and PA, i, and D to get
		the galaxy's absolute units 
		
	Returns
	-------
	
	XY_kpc: (N,2) ndarray
		Contains X and Y coords of all data points with units of kpc
	'''
	delta_RA_deg  = RA  - meta['RA']
	delta_DEC_deg = DEC - meta['DEC']
	PA = np.radians(meta['PA'])
	i  = np.radians(meta['i'])
	# 1: Rotate RA, DEC by PA to get y (major axis direction) and x (minor axis direction)
	x_deg = delta_RA_deg*np.cos(PA)  - delta_DEC_deg*np.sin(PA)
	y_deg = delta_DEC_deg*np.cos(PA) + delta_RA_deg*np.sin(PA)
	# 2: Stretch x values to remove inclination effects
	x_deg = x_deg / np.cos(i)
	# 3: Convert units to kpc
	x_rad = np.radians(x_deg)
	y_rad = np.radians(y_deg)
	x_kpc = x_rad * meta['Dist'] * 1000
	y_kpc = y_rad * meta['Dist'] * 1000
	XY_kpc = np.stack((x_kpc, y_kpc)).T
	return XY_kpc

######################
#    Preprocessing   #
######################

# Caution: these functions are PHANGS specific and may not work in general

def SN_cut(line_df, threshold=3):
	'''
	Replace all spaxels with SN<3 in a certain line with NANs.
	
	Parameters
	----------
	lines_df: hdu list
		A big guy containing all the different emission line data 
		present in PHANGS maps files
		
	threshold: float
		At what S/N do we cut a line? (Defaulted to 3)
		
	Returns
	-------
	lines_df: hdu list
		The same hdu list, but with lines where S/N < threshold
		replaced with np.nan
	'''
	n_lines = 8 # for PHANGS data
	x_max, y_max = line_df[30].data.shape
	for l in range(1, n_lines+1):
		signal = line_df[6*l - 1].data
		noise  = line_df[6*l].data
		too_low = signal <= threshold*noise
		# replace low signals/no signals with NANs.
		for ii in range(x_max):
			for jj in range(y_max):
				if too_low[ii,jj]:
					signal[ii,jj] = np.nan
					noise[ii,jj]  = np.nan
	return line_df
	
def extinction_correction(line_df, wavelengths, R_V=3.1):
	'''
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data 
		present in PHANGS maps files
		
	wavelengths: np.array
		Wavelength of each of the 8 lines in this data cube, in Angstroms.
		
	R_V: float
		The free parameter in ccm89 extinction law. Set (kept) at 3.1.
	
	Returns
	-------
	
	corrected_lines_df: hdu list
		Corrections for all lines using the calibration of ccm89.
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))] # the who's who of line data
	Ha_map = line_df[line_IDs.index('HA6562_FLUX')].data
	Hb_map = line_df[line_IDs.index('HB4861_FLUX')].data
	# To convert balmer decrement to extinction, need these...
	HA_EXT =  ccm89(np.array([6562.8]), 1.0, R_V)[0]
	HB_EXT =  ccm89(np.array([4861.3]), 1.0, R_V)[0]
	Ha_Hb_ratio	 = Ha_map/Hb_map
	balmer_decrement = 2.5*np.log10(Ha_Hb_ratio / 2.86)
	A_V = balmer_decrement/(HB_EXT - HA_EXT) 
	A_V_positive = A_V * (A_V > 0) # sets negatives to zero
	
	# Use this to correct obs and error for each wavelength
	for l in range(len(wavelengths)):
		extinction_at_wav = ccm89(wavelengths[l:l+1], 1, R_V)[0]
		extinction_map = extinction_at_wav*A_V_positive
		# correct signal and noise
		line_df[6+l-1].data	 = line_df[6+l-1].data * 10**(0.4 * extinction_map)
		line_df[6*l].data	 = line_df[6*l].data   * 10**(0.4 * extinction_map)
	
	return line_df

def classify_S2_BPT(line_df):
	'''
	For each spaxel
	specify whether it is SEYFERT, LINER, or SF
	using the diagnostics of Kewley+01 and Kewley+06
	and the S2-BPT diagram.
	
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	Returns
	-------
	
	S2_BPT_classification: np array
		True if in a Hii region
		False if not
		For all spaxels
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	O3Hb = np.log10( line_df[line_IDs.index('OIII5006_FLUX')].data /	 line_df[line_IDs.index('HB4861_FLUX')].data )
	S2Ha = np.log10( (line_df[line_IDs.index('SII6716_FLUX')].data+line_df[line_IDs.index('SII6730_FLUX')].data)/line_df[line_IDs.index('HA6562_FLUX')].data	 )
	is_starburst = O3Hb < ( 0.72/(S2Ha-0.32) + 1.3 )
	return is_starburst & (S2Ha < 0.32)

def classify_N2_BPT(line_df, rule="Kauffmann03"):
	'''
	For each spaxel
	specify whether it is LINER or SF
	using the diagnostic of Kewley+01
	and the N2-BPT diagram.
	
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	Returns
	-------
	N2_BPT_classification: np array
		True if in a Hii region
		False if not
		For all spaxels
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	O3Hb = np.log10( line_df[line_IDs.index('OIII5006_FLUX')].data/line_df[line_IDs.index('HB4861_FLUX')].data )
	N2Ha = np.log10( line_df[line_IDs.index('NII6583_FLUX')].data/line_df[line_IDs.index('HA6562_FLUX')].data	   )
	if rule=='Kewley01':
		is_starburst = O3Hb < 0.61/(N2Ha-0.47) + 1.19
		is_LINER	 = O3Hb >= 0.61/(N2Ha-0.47) + 1.19 # otherwise it's a NAN
	elif rule=='Kauffmann03':
		is_starburst = (O3Hb < 0.61/(N2Ha-0.05) + 1.3) 
		is_LINER	 = (O3Hb > 0.61/(N2Ha-0.05) + 1.3)
	else:
		print("Error: classsify_N2_BPT only works when 'rule' is either 'Kewley01' or 'Kauffmann03'.")
		exit(1)
		return None
	return is_starburst & (N2Ha < 0.05)
	
##########################
#  Spatial Statistics	 #
##########################

def deprojected_distances(RA1, DEC1, RA2=None, DEC2=None, meta=dict()):
	'''
	Computes the deprojected distances between one set of RAs/DECs and
	another, for a known galaxy.
	
	Parameters
	----------
	
	RA1: float, list, or np array-like
		List of (first) RA values. Must be in degrees.
		
	DEC1: float, list, or np array-like
		List of (first) DEC values. Must be in degrees.
		
	RA2: float, list, or np array-like
		(Optional) second list of RA values. Must be in degrees.
		If no argument is provided, then the first list will be used again.
		
	DEC2: float, list, or np array-like
		(Optional) second list of DEC values. Must be in degrees.
		If no argument is provided, then the first list will be used again.	   
	
	meta: dict
		Metadata used to calculate the distances. Must contain:
		PA: float
			Principle Angle of the galaxy, degrees.
		i: float
			inclination of the galaxy along this principle axis, degrees.
		Dist: float
			Distance from this galaxy to Earth, Mpc.
		
	Returns
	-------
	dists: np array
		Array of distances between all RA, DEC pairs provided.
		Units: kpc.
	
	'''
	# Check parameters
	try:
		meta['PA'] 
	except KeyError:
		assert False, "Error: PA not defined for metadata"
	try:
		meta['i'] 
	except KeyError:
		assert False, "Error: i not defined for metadata"
	try:
		meta['Dist'] 
	except KeyError:
		assert False, "Error: Dist not defined for metadata"
	
	# If RA1 and DEC1 are arrays, they must have the same length.
	# If one of them is a float, they must both be floats.
	# You can't supply only one of RA2 and DEC2
	try:
		assert len(RA1) == len(DEC1), "Error: len of RA1 must match len of DEC1"
		RA1 = np.array(RA1)
		DEC1 = np.array(DEC1)
	except TypeError:
		assert type(RA1) == type(DEC1), "Error: type of RA1 must match type of DEC1"  
		# Then cast them to arrays
		RA1 = np.array([RA1])
		DEC1 = np.array([DEC1])
		
	if type(RA2) == type(None):
		RA2 = RA1
	if type(DEC2) == type(None):
		DEC2 = DEC1
	
	try:
		assert len(RA2) == len(DEC2), "Error: len of RA2 must match len of DEC2"
		RA2 = np.array(RA2)
		DEC2 = np.array(DEC2)
	except TypeError:
		assert type(RA2) == type(DEC2), "Error: type of RA2 must match type of DEC2" 
		RA2 = np.array([RA2])
		DEC2 = np.array([DEC2])
	
	# Now onto the maths
	PA = np.radians(meta['PA'])
	i  = np.radians(meta['i'])
	# 1: Rotate RA, DEC by PA to get y (major axis direction) and x (minor axis direction)
	x1 = RA1*np.cos(PA) - DEC1*np.sin(PA)
	y1 = DEC1*np.cos(PA) + RA1*np.sin(PA)
	x2 = RA2*np.cos(PA) - DEC2*np.sin(PA)
	y2 = DEC2*np.cos(PA) + RA2*np.sin(PA)
	# 2: Stretch x values to remove inclination effects
	long_x1 = x1 /np.cos(i)
	long_x2 = x2 /np.cos(i)
	# 3: Compute Euclidean Distances between x1,y1 and x2,y2 to get angular offsets (degrees).
	vec1 = np.stack((y1, long_x1)).T
	vec2 = np.stack((y2, long_x2)).T
	deg_dists = euclidean_distances(vec1, vec2)
	rad_dists = np.radians(deg_dists)
	# 4: Convert angular offsets to kpc distances using D, and the small-angle approximation.
	Mpc_dists = rad_dists * meta['Dist']
	kpc_dists = Mpc_dists * 1000
	
	return kpc_dists
	
def build_error_covariance_matrix(dist_matrix, e_Z, meta=dict()):
	'''
	Build the covariance matrix due to correlated error associated with the 
	measurement of emission lines.
	Assumes PSF of the telescope is a Gaussian.
	
	Parameters
	----------
	
	dist_matrix: (N,N) np.array
		Distances between all pairs of regions.
		
	e_Z: (N,) np.array
		Uncertainty in metallicity for each observation 
		
	meta: dict
		Metadata used to calculate the distances. Must contain:
		D: float
			Distance from this galaxy to Earth, Mpc.
		PSF: float
			Given in Arcseconds, this is the mean seeing for each galaxy (Mean
			value of Table 1 of Emsellem+22 for native resolution for each galaxy: 
			https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.191E/abstract)
	
	Returns
	-------
	cov_matrix: (N,N) np.array
		Covariance matrix for correlated observation errors.
	'''
	# Convert seeing of 0.6'' to kpc, using small angle approximation
	physical_seeing = meta['PSF']*meta['Dist']*1000/ASEC_PER_RAD
	# Convert seeing (FWHM) into a s.d.
	seeing_sd = physical_seeing / (2*np.sqrt(2*np.log(2))) # from	
	# Assume the telescope has a Gaussian PSF:
	correlation_matrix = np.exp(-0.5* (dist_matrix/seeing_sd)**2)
	sd_matrix  = np.diag(e_Z)
	cov_matrix = sd_matrix @ correlation_matrix @ sd_matrix
	return cov_matrix

# Fit a model for the mean trend in metallicity

# Create a semivariogram

# Plot a semivariogram

# Fit a model for the small-scale structure of a galaxy

# Validate a model using N-fold cross-validation

# Krig using data and a model to predict metallicity at an unknown location
