import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pyimzml.ImzMLParser import ImzMLParser
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import signal, stats


class MALDI(object):
	"""A class for reading and simple operations of MALDI imzML Data

	PARAMETERS
	----------
	filename : str
		the filename
	shape : array, shape = [2]
			the size of the two dimensions
	map2D : array, shape = [n_pixels, 2]
			mapping of the flat pixel indices to x and y coordinates
	indices : array, shape = [n_pixels]
			flat pixel indices
	resolution : float
		the relative resolution of any measured mass value
	Range : tuple
		the lower and upper limits of mass values
	n_processes : int
		number of parallel processes
	"""
	def __init__(self, filename, resolution = 2.5e-5, Range = None, n_processes = 1):
		self.filename = filename
		self.file = ImzMLParser(self.filename+".imzML")
		self.shape = np.array([self.file.imzmldict['max count of pixels x'], self.file.imzmldict['max count of pixels y']])
		self.map2D = np.array(self.file.coordinates)[:,:-1]-1		#get x and y indices z index is thrown away as it is non-existent		#minus 1 because some people think counting starts at 1
		self.indices = np.arange(self.map2D.shape[0])		#flat indeces of whole dataset
		self.resolution = resolution
		self.n_processes = n_processes
		if not Range:
			raw = rawMALDI(self.filename, self.resolution, 'dummy', self.n_processes)
			raw.apply_global_range()
			self.Range = raw.Range
		else:
			self.Range = Range

	def get_2D(self, vector):
		"""	return a pixel vector as 2D matrix by applying the 2D mapping

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
		"""retrieve and pretty print the metadata
		"""
		from pyimzml.metadata import Metadata
		import pprint as pp
		metadata = Metadata(self.file.root).pretty()
		pp.pp(metadata)

class rawMALDI(MALDI):
	"""A class for processing raw MALDI data
	
	PARAMETERS
	----------
	file : ImzMLParser object
		Parser for the imzML Data
	data_spectrum : list[lists], shape = [n_pixels, 2]
		contains the mzValues as first element
		and the intensityValues as second element
		each list element is of shape [n_Values]
	"""
	def __init__(self, filename, resolution = 2.5e-5, Range = None, n_processes = 1):
		super().__init__(filename, resolution, Range, n_processes)
		self.data_spectrum = []		#[[mzValues, intensityValues]]
		self.make_data()
		if not self.Range:
			self.apply_global_range()

	def make_data(self):
		"""load the dataspectrum and save the summed intensities as 2D image
		"""
		for pixel in self.indices:
			self.data_spectrum.append(np.array(self.file.getspectrum(pixel)))

	def apply_global_range(self):
		""" set the Range variable to the min and max mz-Values of all pixels
		"""
		mins = []
		maxs = []
		for pixel in self.indices:
			mins.append(np.min(self.data_spectrum[pixel][0]))
			maxs.append(np.max(self.data_spectrum[pixel][0]))
		self.Range = (np.min(mins), np.max(maxs))

	def sumpicture(self):
		""" calculate the sum of the spectra in each pixel and return as array

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
		"""get the index of the nearest measured mz value for a given mz value

		PARAMETERS
		----------
		pixel : int
			the pixel whose data is accesed
		mz : float
			the mz value

		RETURNS
		-------
		index : int
			the index closest to the mz value
		"""
		index = np.abs(self.data_spectrum[pixel][0]-mz).argmin()
		return index

	def getmzint(self, pixel, mz, resolution, suminres = False):
		"""get the intensity of the nearest measured mz value in resolution neighborhood for a given mz value

		PARAMETERS
		----------
		pixel : int
			the pixel whose data is accesed
		mz : float
			the mz value
		resolution: float
			range around the mz-value to search for the nearest value
		suminres: bool
			if true all measured intensities in the resolution will be summed, else just the nearest intensity is returned

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

	def massvec(self, mz, suminres = False, new_resolution = None):
		"""get a vector of the intensitys to the provided mz value in self.resolution or new_resolution in every pixel and optionally sum in the range of the resolution

		PARAMETERS
		----------
		mz : float
			mz value
		suminres: bool
			if true all measured intensities in the resolution will be summed, else just the nearest intensity is returned
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

class binnedMALDI(MALDI):
	"""A class for processing binned MALDI data

	PARAMETERS
	----------
	bins : array, shape [n_bins]
		the binning as in np.arange(min,max,step)
	bincenters : array, shape [n_bins]
		the centers of the bins
	data_histo : array, shape = [n_pixels, n_bins]
		the non-sparse histogram array, defined by bins
	binned_resolution : float
		individual replacement for resolution
	"""
	def __init__(self, filename, resolution = 2.5e-5, Range = None, n_processes = 1, binned_resolution = None, data_histo = None, correlation = None):
		super().__init__(filename, resolution, Range, n_processes)
		if not binned_resolution:
			self.binned_resolution = self.resolution
		else: 
			self.binned_resolution = binned_resolution
		self.bin_data()
		if data_histo is None:
			self.bin_all()
		else:
			self.data_histo = data_histo
		self.correlation = correlation

	def bin_2D(self, data_spectrum, index = 0):
		"""calculate the normalized histogram values acoording to self.bins

		PARAMETERS
		----------
		data_spectrum : rawMALDI.data_spectrum object
			list[lists], shape = [n_pixels, 2]
			contains the mzValues as first element
			and the intensityValues as second element
			each list element is of shape [n_Values]
		index : int
			the index of the pixel to calculate the histogram of

		RETURNS
		-------
		n : array, shape = [self.bins.shape - 1]
			the histogram values
			
		"""
		n,_ = np.histogram(data_spectrum[index][0], bins=self.bins, weights=data_spectrum[index][1])
		return n

	def bin_all(self):
		"""bin the data for all pixel"""
		data_spectrum = rawMALDI(self.filename, self.resolution, self.Range, self.n_processes).data_spectrum
		if self.n_processes >1:
			def bin_2D_parralel(data_specbins):
				""""calculate the normalized histogram values acoording to self.bins

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
			for pixel in self.indices:
				self.data_histo[pixel,:]=self.bin_2D(data_spectrum, pixel)

	def bin_data(self):
		"""bin all data according to self.binned_resolution and self.Range
		"""	
		numbins = math.log(self.Range[1]/self.Range[0], 1+self.binned_resolution*2)
		self.bins = self.Range[0]*(1+self.binned_resolution*2)**np.arange(numbins)
		self.bincenters = np.array([self.bins[i] + (self.bins[i+1]-self.bins[i])/2 for i in range(self.bins.shape[0]-1)])
		self.data_histo = np.zeros((self.map2D.shape[0],len(self.bins)-1))

	def mzindex(self, mz):
		"""return the bin of a mz value
		
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
		"""get a vector of the intensitys to the provided mz value in self.binned_resolution or new_resolution in every pixel

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
				raise ValueError
			else:
				mzbin = np.digitize(mz, self.bins)-1
				massvec = self.data_histo[:,mzbin]			
		else:
			mzbin = np.digitize(mz, self.bins)-1
			massvec = self.data_histo[:,mzbin]			
		return massvec

	def normalize(self, algorithm = 'tic', return_map = False, peaklist = None, inplace = True):
		"""normalize the binned data using specified algorithm on all or selected pixels

		PARAMETERS
		----------
		algorithm : {'tic', 'median'}
			str specifying the algorithm to use, default is 'tic'
		return_map : boolean
			states if the normalization factors for all pixels should be returned as a vector
		peaklist : array-like
			list of peaks relevant for specified normalization method, optional, depending on method
		inplace : bool
			if true self.data_histo is modified and will not be returned

		RETURNS
		-------
		factor : array, shape = self.indices or selected pixels
			the normalization factors for all pixels
		norm_histo : array, shape = self.data_histo.shape
			normalized histo
		"""
		if peaklist:
			raise NotImplementedError
			peaklist = np.digitze(peaklist, self.bins)
		if algorithm == 'tic':
			factor = np.sum(self.data_histo[:,peaklist], axis = 1)/self.data_histo[:,peaklist].shape[0]
		elif algorithm == 'median':
			factor = np.median(self.data_histo[:,peaklist], axis = 1)
		else:
			raise NotImplementedError
		if inplace:
			self.data_histo*=factor[:,np.newaxis]
			if return_map:
				return factor
		else:
			norm_histo = self.data_histo * factor[:,np.newaxis]
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
		method : {'cardinalmeanad', 'meanad', 'medianad', 'snr', 'freq', 'threshold'}, default: 'meanad'
			the method to evaulate the peaks
		localmax : bool
			if true the local maximum in windows of size window is returned as a peak for the used method
		window : int
			the number of consecutive peaks to consider if localmax is used
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
			#from statsmodels import robust		#check documentaation for e.g. normalization, standart is for gaussian distributed noise
			#dev = robust.mad(self.data_histo, axis = 1)[:, np.newaxis]
			#dev = stats.median_abs_deviation(self.data_histo, axis = 1)[:, np.newaxis]
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
			#print(dev.shape)
			if localmax:		#find peaks in every pixels spectra, filtering by distance between neighboring peaks
				indexlist = [None] * self.indices.shape[0]
				for pixel in self.indices:		#KORRR PROBLEM, DASS GGF IN JEDEM PIXEL ANDERS?
					#print(pixel)
					indexlist[pixel], _ = signal.find_peaks(self.data_histo[pixel, :], distance = window//2)		# not identical to R matter::locmax , but almost identical
					indexlist[pixel] = indexlist[pixel][(self.data_histo[pixel, indexlist[pixel]]/dev[pixel]) > threshold]
					peak_histo[pixel, indexlist[pixel]] = self.data_histo[pixel, indexlist[pixel]]
			else:
				peak_histo[(self.data_histo/dev[:,np.newaxis]) > threshold] = self.data_histo[(self.data_histo/dev[:,np.newaxis]) > threshold]

		if inplace:
			self.data_histo = peak_histo
		else:
			return peak_histo

	def peakalign(self, reference = None, reference_params = {'function' : np.mean, 'localmax' : True, 'window' : 2}, tol = None):
		"""align peaks to a reference or the mean spectrum

		PARAMETERS
		----------
		reference : array, shape = [n_peaks]
			reference spectrum, optional
		reference_params : dict
			parameters for reference calculation
		tol : float
			tolerance for peak matching, optional

		RETURNS
		-------
		aligned_histo : array, shape = [n_pixels, n_reference_peaks]
			the array containing the intensities, aligned to the reference peaks or the average spectrum, will only be returned if inplace = False
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
		
			selected = selectedMALDI(self.filename, self.resolution, self.Range, self.n_processes, indices.shape[0])
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
		"""calculate the correlation between the spatial distribution of a refpeak and all other peaks based on peaks in data_histo
		
		PARAMETERS
		----------
		refpeak : float
			mz-value of refpeak for correlation

		RETURNS
		-------
		self.correlation[refindex, :] : array
			the correlation including the refpeak
		"""
		if self.correlation is None:		# implement some case, to not calculate over and over again
			self.calculate_correlation()
		refindex = np.digitize(refpeak, self.bins)-1
		return self.correlation[refindex, :]

	def plot_mean_spec(self, peaklist = None):
		"""make a nice plot of the mean spectrum of the binned data

		PARAMETERS
		----------
		peaklist : array, shape = [n_peaks]
			a list of peaks to mark in the spectrum

		RETURNS
		-------
		mean_spec : array, shape = [n_bins]		
		"""
		mean_spec = np.mean(self.data_histo, axis = 0)
		import matplotlib.pyplot as plt
		plt.figure(figsize = (8,4))
		plt.plot(self.bincenters, mean_spec, linewidth = 1., color = 'green', label = 'average spectrum')
		if peaklist is not None:
			for peak in peaklist[:-1]:
				plt.plot([peak, peak], [0, np.max(mean_spec)], linestyle = '--', color = 'r', alpha = .5)
			plt.plot([peaklist[-1], peaklist[-1]], [0, np.max(mean_spec)], linestyle = '--', color = 'r', alpha = .5, label = 'selected peaks')
		plt.xlabel('$m/z$')
		plt.ylabel('intensity/a.u.')
		plt.legend(loc = 'best')
		plt.tight_layout()


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
		kmeans_image = KMeans(n_clusters=n_clusters).fit(self.data_histo).labels_
		return kmeans_image

	def PCA(self, n_components, standardize = True):
		"""calculate the principal components
		
		PARAMETERS
		----------
		n_components : int
			number of components
		standardize : bool
			if True, the data will be standardized before fitting the PCA, default = True

		RETURNS
		-------
		PCA_image : array, shape = [n_pixels, n_components]
			the pixel array in each component
		PCA_weights : array, shape = [n_components, n_peaks]
			the weights of each components on the original peaks charakterizing their influence on each components
		PCA_evr : array, shape = [n_components]
			the relative explained variance of each component
		"""
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
		TSNE_image = TSNE(n_components = n_components).fit_transform(self.data_histo)
		return TSNE_image

class selectedMALDI(MALDI):
	"""A class for processing MALDI data based on peaklists

	PARAMETERS
	----------
	n_peaks : int
		the number of selected peaks to store
	selected_resolution : float
		relative resolution of peaks
	peakcenters : array, shape = [n_peaks]
		the m/z values of all peaks
	peak_histo : array, shape = [n_pixels, n_peaks]
		the intensities of the peaks in all pixels
	"""
	def __init__(self, filename, resolution = 2.5e-5, Range = None, n_processes = 1, n_peaks = 0, selected_resolution = None, peakcenters = None, peak_histo = None):
		super().__init__(filename, resolution, Range, n_processes)
		if (peakcenters is None) & (peak_histo is None):
			self.peakcenters = np.zeros(n_peaks)
			self.peak_histo = np.zeros((self.indices.shape[0], n_peaks))
		else:
			self.peakcenters = peakcenters
			self.peak_histo = peak_histo
		if not selected_resolution:
			self.selected_resolution = self.resolution

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
		"""get a vector of the intensitys to the provided mz value in every pixel

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
			massvec = self.peak_histo[:,peak]
		return massvec

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
				self.correlation = np.corrcoef(self.peak_histo)
			return self.correlation[refindex, :]
		else:
			raise ValueError
