"""Write dict about used packages and general applications of these classes
"""
import math
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from scipy import signal
from pathos.multiprocessing import ProcessingPool as Pool
import os

class MALDI(object):
	"""A class for reading MALDI-IMS imzML Data and applying simple operations

	PARAMETERS
	----------
	filename : str
		the filename ending in .imzML
	resolution : float
		the relative resolution of any measured mass value
	range_ : tuple
		the lower and upper limits of mass values
	n_processes : int
		number of parallel processes for operations which allow parallelization

	ATTRIBUTES
	----------
	filename : str
		the filename
	resolution : float
		the relative resolution of any measured mass value
	range_ : tuple
		the lower and upper limits of mass values
	n_processes : int
		number of parallel processes for operations which allow parallelization
	file : ImzMLParser object
		the parser for the imzML data by pyimzml
	shape : array, shape = [2]
			the size of the two dimensions
	map2D : array, shape = [n_pixels, 2]
			mapping of the flat pixel indices to x and y coordinates
	indices : array, shape = [n_pixels]
			flat pixel indices

	METHODS
	-------
	get_2D(vector)
		return a pixel vector as 2D matrix by applying the 2D mapping
	print_metadata()
		retrieve and pretty print the metadata included in the imzML file
	"""
	def __init__(self, filename, resolution = 2.5e-5, range_ = None, n_processes = 1):
		self.filename = filename
		if not os.path.isfile(self.filename.split('.')[0] + '.ibd'):
 			raise ValueError('ibd file does not exist')
		self.file = ImzMLParser(self.filename)
		self.shape = np.array([self.file.imzmldict['max count of pixels x'], self.file.imzmldict['max count of pixels y']])
		self.map2D = np.array(self.file.coordinates)[:,:-1]-1		#get x and y indices z index is thrown away as it is non-existent		#minus 1 because some people think counting starts at 1
		self.indices = np.arange(self.map2D.shape[0])		#flat indeces of whole dataset
		self.resolution = resolution
		self.n_processes = n_processes
		self.range_ = range_

	def get_2D(self, vector):
		"""return a pixel vector as 2D matrix by applying the 2D mapping

		PARAMETERS
		----------
		vector : array, shape = [n_pixels]
			a pixel vector

		RETURNS
		-------
		image : array, shape = [self.shape]
			the image matrix
		"""
		image = np.zeros(self.shape)
		image[self.map2D[:,0], self.map2D[:,1]] = vector
		return image

	def print_metadata(self):
		"""retrieve and pretty print the metadata included in the imzML file
		"""
		from pyimzml.metadata import Metadata
		import pprint as pp
		metadata = Metadata(self.file.root).pretty()
		pp.pp(metadata)

class rawMALDI(MALDI):
	"""A class for processing raw MALDI-IMS data
	
	PARAMETERS
	----------
	filename : str
		the filename
	resolution : float
		the relative resolution of any measured mass value
	range_ : tuple
		the lower and upper limits of mass values
	n_processes : int
		number of parallel processes for operations which allow parallelization

	ATTRIBUTES
	----------
	inherited from super class MALDI()
		filename : str
			the filename
		resolution : float
			the relative resolution of any measured mass value
		range_ : tuple
			the lower and upper limits of mass values
		n_processes : int
			number of parallel processes for operations which allow parallelization
		file : ImzMLParser object
			the parser for the imzML data by pyimzml
		shape : array, shape = [2]
				the size of the two dimensions
		map2D : array, shape = [n_pixels, 2]
				mapping of the flat pixel indices to x and y coordinates
		indices : array, shape = [n_pixels]
				flat pixel indices
	data_spectrum : list[lists], shape = [n_pixels, 2]
		contains the mzValues as first element
		and the intensityValues as second element
		each element is of individual length [[mzValues, intensityValues]]

	METHODS
	-------
	sumpicture()
		calculate the sum of the spectrum in each pixel and return the resulting vector
	nearestmzindex(pixel, mz)
		get the index of the nearest measured mz value for a given arbitrary mz value in the given pixel
	getmzint(pixel, mz, resolution, suminres = False)
		get the intensity of the nearest measured mz value in a distance defined by resolution for a given mz value
	massvec(mz, suminres = False, new_resolution = None)
		get a vector of the intensities to the provided mz value in self.resolution or new_resolution in every pixel and optionally sum in the range of the resolution
	normalize(algorithm = 'tic', return_map = False, peaklist = None, inplace = True, thresholds = [None, None])
		normalize the data using specified algorithm
	fit_gauss(positions, sigmas, amps, rel_fitrange, maxvalue_factor = 1.1)
		fit gauss peaks at positions with sigmas and amps starting-parameters
	center_of_mass(massrange = None):
		calculate the center of mass for each spectra in all pixels
	"""
	def __init__(self, filename, resolution = 2.5e-5, range_ = None, n_processes = 1):
		def _make_data():
			"""load the dataspectrum using the imzmlparser into a list of lists
			"""
			for pixel in self.indices:
				self.data_spectrum.append(np.array(self.file.getspectrum(pixel)))

		def _apply_global_range():
			""" set the range_ variable to the min and max mz-Values of all pixels
			"""
			mins = []
			maxs = []
			for pixel in self.indices:
				mins.append(np.min(self.data_spectrum[pixel][0]))
				maxs.append(np.max(self.data_spectrum[pixel][0]))
			self.range_ = (np.min(mins), np.max(maxs))

		super().__init__(filename, resolution, range_, n_processes)
		self.data_spectrum = []		#[[mzValues, intensityValues]]
		_make_data()
		if not self.range_:
			_apply_global_range()

	def sumpicture(self):
		"""calculate the sum of the spectrum in each pixel and return the resulting vector

		RETURNS
		-------
		sumpicture : array, shape = self.indices.shape
			the array including the sum of each pixels spectrum
		"""
		sumpicture = np.zeros(self.indices.shape)
		for pixel in self.indices:
			sumpicture[pixel] = self.data_spectrum[pixel][1].sum()
		return sumpicture

	def nearestmzindex(self, pixel, mz):
		"""get the index of the nearest measured mz value for a given arbitrary mz value in the given pixel

		PARAMETERS
		----------
		pixel : int
			the pixel whose data is accessed
		mz : float
			the mz value

		RETURNS
		-------
		index : int
			the index of the intensity measurement closest to the mz value
		"""
		index = np.abs(self.data_spectrum[pixel][0]-mz).argmin()
		return index

	def getmzint(self, pixel, mz, resolution, suminres = False):
		"""get the intensity of the nearest measured mz value in a distance defined by resolution for a given mz value

		PARAMETERS
		----------
		pixel : int
			the pixel whose data is accesed
		mz : float
			the mz value
		resolution: float
			range around the mz-value to search for the nearest value
		suminres: bool
			if true all measured intensities in the resolution range will be summed, else just the nearest intensity is returned

		RETURNS
		-------
		intensity : int
			the intensity closest to the mz value
		"""

		diffs = np.abs(self.data_spectrum[pixel][0]-mz)
		if suminres:
			minindices = np.where(diffs<resolution*mz)
			if (diffs[minindices] > resolution*mz).any():
				intensity = 0.
			else:
				intensity = np.sum(self.data_spectrum[pixel][1][minindices])
		else:
			minindex = np.argmin(diffs)
			if diffs[minindex] > resolution*mz:
				intensity = 0.
			else:
				intensity = self.data_spectrum[pixel][1][minindex]
		return intensity

	def massvec(self, mz, suminres = True, new_resolution = None):
		"""get a vector of the intensities to the provided mz value in self.resolution or new_resolution in every pixel and optionally sum in the range of the resolution

		PARAMETERS
		----------
		mz : float
			mz value
		suminres: bool
			if true all measured intensities in the resolution range will be summed, else just the nearest intensity is returned, default : True
		new_resolution : float, optional
			resolution to look for nearest peak. self.resolution is used by default

		RETURNS
		-------
		massvec : array, shape = self.shape
		"""
		massvec = np.zeros(self.indices.shape[0])
		if not new_resolution:
			new_resolution = self.resolution
		for pixel in self.indices:
			massvec[pixel] = self.getmzint(pixel, mz, new_resolution, suminres = suminres)
		return massvec

	def normalize(self, algorithm = 'tic', return_map = False, peaklist = None, inplace = True, thresholds = [None, None]):
		"""normalize the data using specified algorithm 

		PARAMETERS
		----------
		algorithm : {'tic', 'ticwindow', 'peaklist', 'tic_perpeak', median'}
			str specifying the algorithm to use, default is 'tic'
		return_map : boolean
			states if the normalization factors for all pixels should be returned as a vector
		peaklist : array-like
			list of peaks relevant for specific normalization method, optional, depending on method
		inplace : bool
			if true self.data_spectrum is modified and will not be returned
		thresholds : tuple
			lower and upper threshold for the tic window

		RETURNS
		-------
		factor : array, shape = self.indices or selected pixels
			the normalization factors for all pixels, only returned if return_map = True
		norm_data_spectrum : list[lists], shape = [n_pixels, 2]
			as data_spectrum but normalized, only returned if inplace = False
		"""
		factor = np.empty(len(self.indices))
		if algorithm == 'tic':
			for pixel in self.indices:
				factor[pixel] = self.data_spectrum[pixel][1].sum()
		elif algorithm == 'ticwindow':
			if thresholds[0] is None:
				thresholds[0] = self.range_[0]
			if thresholds[1] is None:
				thresholds[1] = self.range_[1]
			for pixel in self.indices:
				condition1 = self.data_spectrum[pixel][0] > thresholds[0]
				condition2 = self.data_spectrum[pixel][0] < thresholds[1]
				factor[pixel] = self.data_spectrum[pixel][1][condition1&condition2].sum()
		elif algorithm == 'peaklist':
			for pixel in self.indices:
				for peak in peaklist:
					factor[pixel] += self.getmzint(pixel, peak, self.resolution, suminres= True)
		elif algorithm == 'tic_perpeak':
			for pixel in self.indices:
				factor[pixel] = self.data_spectrum[pixel][1].sum()/len(self.data_spectrum[pixel][1])
		elif algorithm == 'median':		#TODO implement
			raise NotImplementedError
		else:
			raise NotImplementedError
		if inplace:
			for pixel in self.indices:
				self.data_spectrum[pixel][1]/=factor[pixel]
			if return_map:
				return factor
		else:
			norm_data_spectrum = copy.deepcopy(self.data_spectrum)
			for pixel in self.indices:
				norm_data_spectrum[pixel][1]/=factor[pixel]
			if return_map:
				return factor, norm_data_spectrum
			else:
				return norm_data_spectrum

	def fit_gauss(self, positions, sigmas, amps, rel_fitrange, maxvalue_factor = 1.1, rsquared_thresh = .8, retries = 3):
		"""fit gauss peaks at positions with sigmas and amps starting-parameters

		PARAMETERS
		----------
		positions: array, shape = [n_peaks]
			the estimated positions of the peaks, may be single list for all spectra
		sigmas : array, shape = [n_peaks] or [n_peaks, n_pixels]
			the estimated sigma of the peaks
		amps : array, shape = [n_peaks] or [n_peaks, n_pixels]
			the estimated amplitude of the peaks
		rel_fitrange : float
			the relative range around peaks for fitting
		maxvalue_factor : float
			the factor with which the fit amplitude is allowed to exceed the maximal value in the data, default : 1.1
		rsquared_thresh : float
			a threshold for the rsquared value of the fit to accept the fit. If the fit is not accepted, it will be retried retries times
		retries : int
			the number of retries to do if the rsquared threshold is not satisfied

		RETURNS
		-------
		chi_res : array, shape = [n_pixels]
			the reduced chi squares of the fits
		amps : array, shape = [n_peaks, n_pixels]
			the resulting amplitude of each gauss function
		x0 : array, shape = [n_peaks, n_pixels]
			the resulting position of each gauss function
		sigma : array, shape = [n_peaks, n_pixels]
			the resulting sigma of each gauss function
		amps_err : array, shape = [n_peaks, n_pixels]
			the resulting amplitude error of each gauss function
		x0_err : array, shape = [n_peaks, n_pixels]
			the resulting position error of each gauss function
		sigma_err : array, shape = [n_peaks, n_pixels]
			the resulting sigma error of each gauss function
		"""
		from lmfit import models,Model
		def _gausss(x, amplitude, center, sigma):
			return amplitude*np.exp(-((x-center)/sigma)**2/2)
		def _fit_pixelwise(param_list):
			rsquared = param_list[4] -1
			count = 0
			while rsquared_not_reached:
				fit = param_list[3].fit(x = param_list[0], params = param_list[2], data = param_list[1])
				fit_sampling = fit.eval(fit.params, x = param_list[0])
				rsquared = 1 - np.sum((param_list[1]-fit_sampling)**2)/np.sum((param_list[1]-np.mean(param_list[1]))**2)

				if rsquared > param_list[4]:
					return [fit.redchi	, fit.result.params['amplitude'].value, fit.result.params['center'].value, fit.result.params['sigma'].value, fit.result.params['amplitude'].stderr, fit.result.params['center'].stderr, fit.result.params['sigma'].stderr, rsquared]
				elif count < param_list[5]:
					rsquared_not_reached = True
					count += 1
				else:
					return [None, None, None, None, None, None, None, rsquared]


		fitrange = positions*rel_fitrange
		n_peaks = positions.shape[0]
		amp = np.zeros((n_peaks, self.indices.shape[0]))
		x0 = np.zeros((n_peaks, self.indices.shape[0]))
		sigma = np.zeros((n_peaks, self.indices.shape[0]))
		amps_err = np.zeros((n_peaks, self.indices.shape[0]))
		x0_err = np.zeros((n_peaks, self.indices.shape[0]))
		sigma_err = np.zeros((n_peaks, self.indices.shape[0]))

		maxs = []
		for pixel in self.indices:
			maxs.append(np.max(self.data_spectrum[pixel][1]))
		maxvalue = np.max(maxs)*maxvalue_factor
		center_range = positions*self.resolution
		chi_res = np.zeros((n_peaks, self.indices.shape[0]))
		failed_pixels = [[] for _ in range(n_peaks)]

		for peak in range(n_peaks):
			if positions[peak] < self.range_[0]:
				print('peak ' + str(peak) + ' out of range')
				for pixel in self.indices:
					chi_res[peak,pixel] = None
					amp[peak,pixel] = None
					x0[peak,pixel] = None
					sigma[peak,pixel] = None
					amps_err[peak,pixel] = None
					x0_err[peak,pixel] = None
					sigma_err[peak,pixel] = None
				continue
			print('\nadding model for peak ', peak)
			model = Model(_gausss)
			params = model.make_params()
			params['amplitude'].set(amps[peak], min = maxvalue*1e-8, max = maxvalue)
			params['center'].set(positions[peak], min = positions[peak] - center_range[peak], max = positions[peak] + center_range[peak])
			params['sigma'].set(sigmas[peak], min = 1e-32, max = sigmas[peak]*5)
			param_list = []
			for pixel in self.indices:
				lower = self.nearestmzindex(pixel, positions[peak] - fitrange[peak])
				higher = self.nearestmzindex(pixel, positions[peak] + fitrange[peak])
				if higher - lower < 3:#lower == higher:
					failed_pixels[peak].append(pixel)
					chi_res[peak,pixel] = None
					amp[peak,pixel] = None
					x0[peak,pixel] = None
					sigma[peak,pixel] = None
					amps_err[peak,pixel] = None
					x0_err[peak,pixel] = None
					sigma_err[peak,pixel] = None
					continue
				print('peak ' + str(peak) + ' not measured in pixels ' + str(failed_pixels[peak]), end = '\r')
				param_list.append([self.data_spectrum[pixel][0][lower:higher], self.data_spectrum[pixel][1][lower:higher], params, model, rsquared_thresh, retries])
			if len(param_list)>0:
				pool = Pool(self.n_processes)
				fit_results = pool.map(_fit_pixelwise, param_list)
				pool.close()
				pool.join()
				pool.terminate()
				pool.restart()
				fit_results = np.array(fit_results)

				nonnan = np.nonzero(~np.isnan(chi_res[peak,:]))[0]
				chi_res[peak,nonnan] = fit_results[:,0]
				amp[peak,nonnan] = fit_results[:,1]
				x0[peak,nonnan] = fit_results[:,2]
				sigma[peak,nonnan] = fit_results[:,3]
				amps_err[peak,nonnan] = fit_results[:,4]
				x0_err[peak,nonnan] = fit_results[:,5]
				sigma_err[peak,nonnan] = fit_results[:,6]
				rsquared[peak,nonnan] = fit_results[:,7]
		return rsquared, chi_res, amp, x0, sigma, amps_err, x0_err, sigma_err

	def center_of_mass(self, massrange = None):
		"""calculate the center of mass for each spectra in all pixels

		PARAMETERS
		----------
		massrange : tuple
			lower and upper limit of the mass range to consider in center of mass calculation, optional

		RETURN
		------
		com : array, shape = self.indices.shape[0]
			the center of mass for all pixels
		"""
		if massrange is None:
			massrange = self.range_
		com = np.zeros(self.indices.shape[0])
		for pixel in self.indices:
			massindices = np.nonzero((self.data_spectrum[pixel][0]>massrange[0])&(self.data_spectrum[pixel][0]<massrange[1]))
			com[pixel] = np.sum(self.data_spectrum[pixel][0][massindices]*self.data_spectrum[pixel][1][massindices])/np.sum(self.data_spectrum[pixel][1][massindices])
		return com

class binnedMALDI(MALDI):
	"""A class for processing binned MALDI-IMS data, which allows to load already binned data

	PARAMETERS
	----------
	filename : str
		the filename
	resolution : float
		the relative resolution of any measured mass value
	range_ : tuple
		the lower and upper limits of mass values
	n_processes : int
		number of parallel processes for operations which allow parallelization
	binned_resolution : float
		individual replacement for resolution, optional
	data_histo : array, shape = [n_pixels, n_bins]
		the non-sparse histogram array, optional, will be calculated fresh if not provided
	data_spectrum : list[lists], shape = [n_pixels, 2]
		contains the mzValues as first element
		and the intensityValues as second element
		each element is of individual length [[mzValues, intensityValues]]
		optional, will be calculated fresh if not provided
	correlation : array,shape = [n_bins, n_bins]
		correlation between the intensity distribution in the pixels, optional, can be calculated by calling calculate_correlation()

	ATTRIBUTES
	----------
	inherited from super class MALDI()
		filename : str
			the filename
		resolution : float
			the relative resolution of any measured mass value
		range_ : tuple
			the lower and upper limits of mass values
		n_processes : int
			number of parallel processes for operations which allow parallelization
		file : ImzMLParser object
			the parser for the imzML data by pyimzml
		shape : array, shape = [2]
				the size of the two dimensions
		map2D : array, shape = [n_pixels, 2]
				mapping of the flat pixel indices to x and y coordinates
		indices : array, shape = [n_pixels]
				flat pixel indices
	bins : array, shape [n_bins + 1]
		the threshold values of the binning of the data in the m/z dimension including lower and upper limits
	bincenters : array, shape [n_bins]
		the centers of the bins
	data_histo : array, shape = [n_pixels, n_bins]
		the non-sparse histogram array, defined by bins
	data_spectrum : list[lists], shape = [n_pixels, 2]
		contains the mzValues as first element
		and the intensityValues as second element
		each element is of individual length [[mzValues, intensityValues]]
	binned_resolution : float
		optional individual replacement for resolution

	METHODS
	-------
	mzindex(mz)
		return the bin of an mz value
	massvec(mz, new_resolution = None)
		get a vector of the intensities to the provided mz value in self.binned_resolution or new_resolution in every pixel
	normalize(algorithm = 'tic', return_map = False, peaklist = None, inplace = True:
		normalize the binned data using specified algorithm on all or selected peaks
	peakpick(threshold, method = 'meanad', localmax = True, window = 5, inplace = True)
		pick the peaks which return a snr value greater than threshold after evaluation using method
	peakalign(reference = None, reference_params = {'fnction' : np.mean, 'localmax' : True, 'window' : 2}, tol = None):
		align peaks to a reference or the mean spectrum and return a selectedMALDI object
	calculate_correlation()
		calculate the correlation of all entries in self.data_histo
	correlatepeaks(refpeak)
		return the correlation between the spatial distribution of a reference peak and all other peaks based on peaks in data_histo
	plot_mean_spec(peaklist = None)
		make a plot of the mean spectrum of the binned data
	kmeans(n_clusters)
		calculate n_clusters using a kmeans algorithm
	PCA(n_components, standardize = True)
		calculate the principal components
	TSNE(n_components)
		calculate the t-distributed stochastic neighbor embedding
	"""
	def __init__(self, filename, resolution = 2.5e-5, range_ = None, n_processes = 1, binned_resolution = None, data_histo = None, data_spectrum = None, correlation = None):

		def _bin_all(data_spectrum):
			"""bin the data for all pixel"""
			if self.n_processes >1:
				def bin_2D_parralel(data_specbins):
					""""calculate the normalized histogram values according to self.bins parallelized, scaled arbitrary

					PARAMETERS
					----------
					data_specbins : list[lists], shape = [n_pixels, 2]
					contains the [mzValues, intensityValues] as first element
					and the bins as second element

					RETURNS
					-------
					n : array, shape = [bins.shape - 1]
						the histogram values
					"""
					n,_ = np.histogram(data_specbins[0][0], bins=data_specbins[1], weights=data_specbins[0][1])
					return n
				self.data_specbins = []
				for pixel in self.indices:
					self.data_specbins.append([data_spectrum[pixel], self.bins])
				pool = Pool(self.n_processes)
				self.data_histo = pool.map(bin_2D_parralel, self.data_specbins)
				pool.close()
				pool.join()
				pool.terminate()
				pool.restart()
				self.data_histo = np.array(self.data_histo)
			else:
				def bin_2D(data_spectrum, index = 0):
					"""calculate the histogram values according to self.bins, scaled arbitrary

					PARAMETERS
					----------
					data_spectrum : rawMALDI.data_spectrum object
					contains the mzValues as first element
					and the intensityValues as second element
					each element is of individual length [[mzValues, intensityValues]]
					index : int
						the index of the pixel to calculate the histogram of

					RETURNS
					-------
					n : array, shape = [self.bins.shape - 1]
						the histogram values
						
					"""
					n,_ = np.histogram(data_spectrum[index][0], bins=self.bins, weights=data_spectrum[index][1])
					return n
				for pixel in self.indices:
					self.data_histo[pixel,:]= bin_2D(data_spectrum, pixel)

		def _bin_data():
			"""create bins, bincenters and empty data_histo data according to self.binned_resolution and self.range_
			"""	
			numbins = math.log(self.range_[1]/self.range_[0], 1+self.binned_resolution*2)
			self.bins = self.range_[0]*(1+self.binned_resolution*2)**np.arange(numbins)
			self.bincenters = np.array([self.bins[i] + (self.bins[i+1]-self.bins[i])/2 for i in range(self.bins.shape[0]-1)])
			self.data_histo = np.zeros((self.map2D.shape[0],len(self.bins)-1))

		def _apply_global_range(data_spectrum):
			""" set the range_ variable to the min and max mz-Values of all pixels

				PARAMETERS
				----------
				data_spectrum : rawMALDI.data_spectrum object
				contains the mzValues as first element
				and the intensityValues as second element
				each element is of individual length [[mzValues, intensityValues]]
			"""
			mins = []
			maxs = []
			for pixel in self.indices:
				mins.append(np.min(data_spectrum[pixel][0]))
				maxs.append(np.max(data_spectrum[pixel][0]))
			self.range_ = (np.min(mins), np.max(maxs))

		super().__init__(filename, resolution, range_, n_processes)
		if not self.range_:
			if data_spectrum is None:
				data_spectrum = rawMALDI(self.filename, self.resolution, self.range_, self.n_processes).data_spectrum
			_apply_global_range(data_spectrum)
		if not binned_resolution:
			self.binned_resolution = self.resolution
		else: 
			self.binned_resolution = binned_resolution
		_bin_data()
		if data_histo is None:
			if data_spectrum is None:
				data_spectrum = rawMALDI(self.filename, self.resolution, self.range_, self.n_processes).data_spectrum
			_bin_all(data_spectrum)
		else:
			self.data_histo = data_histo
		self.correlation = correlation

	def mzindex(self, mz):
		"""return the bin of an mz value
		
		PARAMETERS
		----------
		mz : float
			the mz value

		RETURNS
		-------
		index : int
			the index of the bin
		"""
		return np.digitize(mz, self.bins)-1

	def massvec(self, mz, new_resolution = None):
		"""get a vector of the intensities to the provided mz value in self.binned_resolution or new_resolution in every pixel

		PARAMETERS
		----------
		mz : float
			mz value
		new_resolution : float, optional
			resolution to look for nearest peak. self.binned_resolution is used by default

		RETURNS
		-------
		massvec : array, shape = self.shape
		"""
		massvec = np.zeros(self.indices.shape[0])
		if 	new_resolution:
			if 	new_resolution>self.binned_resolution:
				mzbins = np.digitize(np.arange(mz-mz*new_resolution, mz+mz*new_resolution, self.binned_resolution*2), self.bins)-1
				massvec = np.sum(self.data_histo[:,mzbins], axis = 1)
			elif new_resolution<self.binned_resolution:
				raise ValueError('new_resolution mus be greater or equal to self.binned_resolution')
			else:
				mzbin = np.digitize(mz, self.bins)-1
				massvec = self.data_histo[:,mzbin]			
		else:
			mzbin = np.digitize(mz, self.bins)-1
			massvec = self.data_histo[:,mzbin]			
		return massvec

	def normalize(self, algorithm = 'tic', return_map = False, peaklist = None, inplace = True):
		"""normalize the binned data using specified algorithm on all or selected peaks

		PARAMETERS
		----------
		algorithm : {'tic', 'median'}
			str specifying the algorithm to use, default : 'tic'
		return_map : boolean
			states if the normalization factors for all pixels should be returned as a vector, default : False
		peaklist : array-like
			list of peaks relevant for specific normalization method, optional, depending on method
		inplace : bool
			if true self.data_histo is modified and will not be returned, default : True

		RETURNS
		-------
		factor : array, shape = self.indices
			the normalization factors for all pixels
		norm_histo : array, shape = self.data_histo.shape
			normalized histo
		"""
		if peaklist:
			peaklist = np.digitze(peaklist, self.bins)
		else:
			peaklist = np.arange(len(self.bins)-1)
		if algorithm == 'tic':
			factor = np.sum(self.data_histo[:,peaklist], axis = 1)/self.data_histo[:,peaklist].shape[0]
		elif algorithm == 'median':
			factor = np.median(self.data_histo[:,peaklist], axis = 1)
		else:
			raise NotImplementedError
		if inplace:
			self.data_histo/=factor[:,np.newaxis]
			if return_map:
				return factor
		else:
			norm_histo = self.data_histo / factor[:,np.newaxis]
			if return_map:
				return factor, norm_histo
			else:
				return norm_histo

	def peakpick(self, threshold, method = 'meanad', localmax = True, window = 5, inplace = True):
		"""pick the peaks which return a snr value greater than threshold after evaluation using method

		PARAMETERS
		----------
		threshold : float
			threshold of value for the method for keeping peaks
		method : {'cardinalmeanad', 'meanad', 'medianad', 'snr', 'freq', 'threshold'}, default : 'meanad'
			the method to evaulate the peaks
		localmax : bool
			if true, before snr calculation, peak candidates are selected as
			the local maximum in ranges of size window
		window : int
			the number of consecutive bins to consider if localmax is used
		inplace : bool
			if true self.data_histo is modified and nothing will be returned

		RETURNS
		-------
		peak_histo : array, shape = self.data_histo.shape
			the array of intensities after peackpicking, will only be returned if inplace = False
		"""
		peak_histo = np.zeros(self.data_histo.shape)
		if method == 'meanad':		#maen absolutte deviation (from the mean)
			dev = np.mean(np.abs(self.data_histo - np.mean(self.data_histo, axis = 1)[:, np.newaxis]), axis = 1)
			nextstep = 'snr'
		elif method == 'cardinalmeanad':		#mean absolute deviation from the mean of the differences between neighboring signals
			dev = np.mean(np.abs(self.data_histo - np.mean(np.diff(self.data_histo, axis = 1), axis = 1)[:, np.newaxis]), axis = 1)
			nextstep = 'snr'
		elif method == 'medianad':		#median absolute deviation (from the mean)
			dev = np.median(np.abs(self.data_histo - np.mean(self.data_histo, axis = 1)[:, np.newaxis]), axis = 1)
			nextstep = 'snr'
		elif method == 'threshold':
			for pixel in self.indices:
				peak_histo[pixel, self.data_histo[pixel, :] > threshold] = self.data_histo[pixel, self.data_histo[pixel, :] > threshold]
		elif method == 'freq':
			n, _ = np.count_nonzero(self.data_histo, axis = 0)
			n /= self.data_histo.shape[0]
			peak_histo[pixel, n > threshold] = self.data_histo[pixel, n > threshold]
		else:
			raise NotImplementedError

		if nextstep == 'snr':
			if localmax:		#find peaks in every pixels spectra, filtering by distance between neighboring peaks
				indexlist = [None] * self.indices.shape[0]
				for pixel in self.indices:
					#print(pixel)
					indexlist[pixel], _ = signal.find_peaks(self.data_histo[pixel, :], distance = window//2)		# not identical to R matter::locmax , but similar
					indexlist[pixel] = indexlist[pixel][(self.data_histo[pixel, indexlist[pixel]]/dev[pixel]) > threshold]
					peak_histo[pixel, indexlist[pixel]] = self.data_histo[pixel, indexlist[pixel]]
			else:
				peak_histo[(self.data_histo/dev[:,np.newaxis]) > threshold] = self.data_histo[(self.data_histo/dev[:,np.newaxis]) > threshold]

		if inplace:
			self.data_histo = peak_histo
		else:
			return peak_histo

	def peakalign(self, reference = None, reference_params = {'function' : np.mean, 'localmax' : True, 'window' : 2}, tol = None):
		"""align peaks to a reference or the mean spectrum and return a selectedMALDI object

		PARAMETERS
		----------
		reference : array, shape = [n_peaks]
			reference spectrum, optional, if None reference spectrum will be calculated by reference parameters
		reference_params : dict
			parameters for reference calculation {'function' : method, 'localmax' : bool, 'window' : int}
			function : method which takes self.data_histo and axis = 0 as input arguments
			and returns a 1-dimensional array
			e.g. np.mean, np.median, ...
			localmax : if true, before alignment, peak candidates are selected as
			the local maximum in ranges of size window
			window : int
				the number of consecutive bins to consider if localmax is used
			default : {'function' : np.mean, 'localmax' : True, 'window' : 2}
		tol : float
			tolerance for peak matching, optional

		RETURNS
		-------
		selected : selectedMALDI object
			a object only containing the selected and aligned peaks
		"""

		if not tol:
			tol = self.binned_resolution
		if not reference:
			reference = reference_params['function'](self.data_histo, axis = 0)
		#print(reference.shape)
		if reference_params['localmax']:		#find peaks in every pixels spectra, filtering by distance between neighboring peaks
			indices, _ = signal.find_peaks(reference, distance = reference_params['window']//2)		# not identical to R matter::locmax , but almost identical
			# Cardinal does one more step of summing/averaging around the peaks, but it does only do this on the intensity, so I do not see the reason behind that
			#print(indices)
			prominence, left, right = signal.peak_prominences(reference, indices)		#calculate prominence for peak border estimation
		
			selected = selectedMALDI(self.filename, self.resolution, self.range_, self.n_processes, indices.shape[0])
			selected.peakcenters = np.zeros(indices.shape[0])
			for i in range(indices.shape[0]):
				lef = left[i] + (indices[i]-left[i])//2		#set the peak boarder at half of the prominence bases for peak i
				rig = right[i] - (right[i]-indices[i])//2
				selected.peakcenters[i] = np.sum(self.bincenters[lef:rig]*reference[lef:rig])/np.sum(reference[lef:rig])		#calculate the peak center by weighting the mz by intensity
		else:
			selected.peakcenters = reference
		#print(selected.peakcenters.shape)
		plusminus = int(tol//self.binned_resolution)		#tolerance in bins
		for i, binn in zip(np.arange(selected.peakcenters.shape[0]), np.digitize(selected.peakcenters, self.bins)-1):
			if plusminus>0:
				selected.peak_histo[:,i] = np.sum(self.data_histo[:, binn-plusminus:binn+plusminus], axis = 1)
			else:
				selected.peak_histo[:,i] = self.data_histo[:, binn]
		return selected

	def calculate_correlation(self):
		"""calculate the correlation of all entries in self.data_histo
		"""
		self.correlation = np.corrcoef(np.transpose(self.data_histo))

	def correlatepeaks(self, refpeak):
		"""return the correlation between the spatial distribution of a reference peak and all other peaks based on peaks in data_histo
		
		PARAMETERS
		----------
		refpeak : float
			mz-value of refpeak for correlation

		RETURNS
		-------
		self.correlation[refindex, :] : array
			the correlation including the refpeak
		"""
		if self.correlation is None:
			self.calculate_correlation()
		refindex = np.digitize(refpeak, self.bins)-1
		return self.correlation[refindex, :]

	def plot_mean_spec(self, peaklist = None):
		"""make a plot of the mean spectrum of the binned data

		PARAMETERS
		----------
		peaklist : array, shape = [n_peaks]
			a list of peaks to mark in the spectrum
		"""
		import matplotlib.pyplot as plt
		mean_spec = np.mean(self.data_histo, axis = 0)
		plt.figure(figsize = (7.5,4))
		plt.plot(self.bincenters, mean_spec, linewidth = 1., color = 'green', label = 'average spectrum')
		if peaklist is not None:
			for peak in peaklist[:-1]:
				plt.plot([peak, peak], [0, np.max(mean_spec)], linestyle = '--', color = 'r', alpha = .5)
			plt.plot([peaklist[-1], peaklist[-1]], [0, np.max(mean_spec)], linestyle = '--', color = 'r', alpha = .5, label = 'selected peaks')
		plt.xlabel('$m/z$')
		plt.ylabel('intensity/a.u.')
		plt.legend(loc = 'best')
		plt.tight_layout()
		plt.show()

	def kmeans(self, n_clusters):
		"""calculate n_clusters using a kmeans algorithm
		
		PARAMETERS
		----------
		n_cluster : int
			number of clusters

		RETURNS
		-------
		kmeans_image : array, shape = [n_pixels]
			the pixel array with the value of the cluster
		"""
		from sklearn.cluster import KMeans
		kmeans_image = KMeans(n_clusters=n_clusters).fit(self.data_histo).labels_
		return kmeans_image

	def PCA(self, n_components, standardize = True):
		"""calculate the principal components
		
		PARAMETERS
		----------
		n_components : int
			number of components
		standardize : bool
			if True, the data will be standardized before fitting the PCA, default : True

		RETURNS
		-------
		PCA_image : array, shape = [n_pixels, n_components]
			the pixel array in each component
		PCA_weights : array, shape = [n_components, n_peaks]
			the weights of each components on the original peaks charakterizing their influence on each components
		PCA_evr : array, shape = [n_components]
			the relative explained variance of each component
		"""
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import StandardScaler
		if standardize:
			scaler = StandardScaler().fit(self.data_histo)
			pca = PCA(n_components = n_components).fit(scaler.transform(self.data_histo))
		else:
			pca = PCA(n_components = n_components).fit(self.data_histo)
		PCA_image = pca.transform(self.data_histo)
		PCA_weights = pca.components_
		PCA_evr = pca.explained_variance_ratio_
		return PCA_image, PCA_weights, PCA_evr

	def TSNE(self, n_components):
		"""calculate the t-distributed stochastic neighbor embedding
		
		PARAMETERS
		----------
		n_components : int
			number of components

		RETURNS
		-------
		TSNE_image : array, shape = [n_pixels, n_components]
			the pixel array in each component
		"""
		from sklearn.manifold import TSNE
		TSNE_image = TSNE(n_components = n_components).fit_transform(self.data_histo)
		return TSNE_image

class selectedMALDI(MALDI):
	"""A class for processing MALDI data based on peaklists

	PARAMETERS
	----------
	filename : str
		the filename
	resolution : float
		the relative resolution of any measured mass value
	range_ : tuple
		the lower and upper limits of mass values
	n_processes : int
		number of parallel processes for operations which allow parallelization
	n_peaks : int
		the number of selected peaks
	selected_resolution : float
		individual replacement for resolution, optional
	peakcenters : array, shape = [n_peaks]
		the m/z values of the selected peaks, optional
	peak_histo : array, shape = [n_pixels, n_peaks]
		the intensities of the selected peaks in all pixels, optional
	correlation : array,shape = [n_bins, n_peaks]
		correlation between the intensity distribution in the pixels, optional, can be calculated by calling calculate_correlation()

	ATTRIBUTES
	----------
	inherited from super class MALDI()
		filename : str
			the filename
		resolution : float
			the relative resolution of any measured mass value
		range_ : tuple
			the lower and upper limits of mass values
		n_processes : int
			number of parallel processes for operations which allow parallelization
		file : ImzMLParser object
			the parser for the imzML data by pyimzml
		shape : array, shape = [2]
				the size of the two dimensions
		map2D : array, shape = [n_pixels, 2]
				mapping of the flat pixel indices to x and y coordinates
		indices : array, shape = [n_pixels]
				flat pixel indices
	selected_resolution : float
		individual replacement for resolution, optional
	peakcenters : array, shape = [n_peaks]
		the m/z values of the selected peaks, optional
	peak_histo : array, shape = [n_pixels, n_peaks]
		the intensities of the selected peaks in all pixels, optional

	METHODS
	-------
	nearestpeak(mz)
		get the index of the nearest measured mz value for a given mz value
	massvec(mz)
		get a vector of the intensities to the provided mz value in every pixel
	correlatepeaks(refpeak)
		calculate the correlation between the spatial distribution of a refpeak and all other peaks based on peaks in peak_histo
	"""
	def __init__(self, filename, resolution = 2.5e-5, range_ = None, n_processes = 1, n_peaks = 0, selected_resolution = None, peakcenters = None, peak_histo = None, correlation = None):
		super().__init__(filename, resolution, range_, n_processes)
		if (peakcenters is None) & (peak_histo is None):
			self.peakcenters = np.zeros(n_peaks)
			self.peak_histo = np.zeros((self.indices.shape[0], n_peaks))
		else:
			self.peakcenters = peakcenters
			self.peak_histo = peak_histo
		if not selected_resolution:
			self.selected_resolution = self.resolution
		self.correlation = correlation

	def nearestpeak(self, mz):
		"""get the index of the nearest measured mz value for a given mz value

		PARAMETERS
		----------
		mz : float
			the mz value

		RETURNS
		-------
		index : int
			the index closest to the mz value
		"""
		diffs = np.abs(self.peakcenters-mz)
		index = diffs.argmin()
		if diffs[index] > self.selected_resolution:
			index = None
		return index

	def massvec(self, mz):
		"""get a vector of the intensities to the provided mz value in every pixel

		PARAMETERS
		----------
		mz : float
			mz value

		RETURNS
		-------
		massvec : array, shape = self.shape
		"""
		massvec = np.zeros(self.indices.shape[0])
		peak = self.nearestpeak(mz)
		if peak is not None:
			massvec = self.peak_histo[:,peak]		#TODO check if this works with returned None values
		return massvec

	def calculate_correlation(self):
		"""calculate the correlation of all entries in self.peak_histo
		"""
		self.correlation = np.corrcoef(np.transpose(self.peak_histo))

	def correlatepeaks(self, refpeak):
		"""calculate the correlation between the spatial distribution of a refpeak and all other peaks based on peaks in peak_histo
		
		PARAMETERS
		----------
		refpeak : float
			mz-value of refpeak for correlation

		RETURNS
		-------
		self.correlation[refindex, :] : array
			the correlation including the refpeak itself
		"""
		refindex = self.nearestpeak(refpeak)
		if refindex:
			if not self.correlation:
				self.calculate_correlation()
			return self.correlation[refindex, :]
		else:
			raise ValueError('No peak stored in resolution range of mz-value')

	def mask_on(self, n_final_clusters, init_clusters, random_state = None):
		"""

		PARAMETERS
		----------
		init_cluster : array of int, shape = [n_pixels]
			clusters marked by increasing integer values >0
		"""
		from sklearn.cluster import KMeans
		if init_clusters:
			#mask = np.repeat(init_clusters, self.peak_histo.shape[1], axis = 1)
			n_init_clusters =  len(np.unique(init_clusters))
			init_cluster_values = []#np.zeros((n_init_clustersself.peak_histo.shape[1]))
			for i in np.arange(1,n_init_clusters):
				init_cluster_values.append(np.mean(peak_histo[init_clusters == i], axis = 0))
			if n_final_clusters > n_init_clusters:
				kmeans_remaining = KMeans(n_clusters = n_final_clusters - n_init_clusters, random_state = random_state).fit(self.peak_histo[init_clusters==0, :])
				print(len(init_cluster_values))
				print(kmeans_remaining.cluster_ccenters_.shape)
				kmeans_final = KMeans(n_clusters = n_final_cluster, init = np.append(init_cluster_values, kmeans_remaining.cluster_ccenters_), random_state = random_state).fit(self.peak_histo)
			elif n_final_clusters == n_init_clusters:
				kmeans_final = KMeans(n_clusters = n_final_cluster, init = init_cluster_values, random_state = random_state).fit(self.peak_histo)
		else:
			kmeans_final = KMeans(n_clusters = n_final_cluster, random_state = random_state).fit(self.peak_histo)
		image = kmeans_final.predict(self.peak_histo)
		return image, kmeans_final

