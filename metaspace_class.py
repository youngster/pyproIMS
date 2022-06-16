import pandas as pd
import numpy as np
from metaspace import SMInstance
class metaspace(object):
	"""A class for retrieving and filtering annotations on the METASPACE platform

	PARAMETERS
	----------
	dataset_names : [str]
		list of names of datasets (e.g. timestamp)
	databases : [str]
		{'HMDB', 'LipidMaps', 'SwissLipids'}
		list of identifiers of the databases for annotations
	max_fdr : float
		maximal false detection rate for dataset loading
	proj_id : str
		id of the project, optional
	api_key : str
		key for accessing the data

	ATTRIBUTES
	----------
	dataset_names : [str]
		list of names of datasets (e.g. timestamp)
	databases : [str]
		{'HMDB', 'LipidMaps', 'SwissLipids'}
		list of identifiers of the databases for annotations
	max_fdr : float
		maximal false detection rate for dataset loading
	sm : SMInstance
		instance for accessing METASPACE data
	proj_id : str
		id of the project
	database_ids : dict
		lookup_table for the ids of databases
		currently 'HMDB', 'LipidMaps', 'SwissLipids' are available
		others may be looked up on METASPACE URLs
	base_url : [str]
		strings neeeded for the URL pattern to access METASPACE data
	data : pd.DataFrame
		Dataframe containing all datasets and their information
		will be created by calling get_datasets()

	METHODS
	-------
	get_urls(self)
		return a dict including the urls to acces the datasets via the browser interface of METASPACE
	get_datasets(drop_duplicates = False, sort_by = None)
		get all selected datasets from METASPACE
	get_dataset(dataset_name, database_id)
		get a single dataset from METASPACE
	filter_by_group(mygroups)
		filter the dataset by matching the molecule group in the moleculeNames
	filter_by(parameters = ['msm'], values = [.8], operators = ['>'])
		filter the dataset by comparing the given set of parameters and values using the operators
	filter_neighboring_mzs(rrange = 10e-6)
		remove the second of two neighboring annotations if mz value is inside the range - works only if sorted by mz
	get_image(dataset_name, database, mz)
		get a dataset image to adduct
	"""
	def __init__(self, dataset_names, databases = ['SwissLipids'], max_fdr = .05, proj_id = None, api_key = None):
		def _create_instance(api_key):
			"""create a SMInstance do load annotation data from METASPACE

			PARAMETERS
			----------
			api_key : str
				key for accessing the data

			RETURNS
			-------
			sm : SMInstance
				instance for accessing METASPACE data
			"""
			sm = SMInstance(host = 'https://metaspace2020.eu', api_key = api_key)
			if not sm.logged_in():
				import getpass
				api_key = getpass.getpass(prompt='API key: ', stream=None)
				sm.login(api_key=api_key)
			return sm

		self.dataset_names = dataset_names
		self.databases = databases
		self.max_fdr = max_fdr
		self.sm = _create_instance(api_key)
		self.proj_id = proj_id		#TODO still needed?
		self.database_ids = {'HMDB' : 22, 'LipidMaps' : 24, 'SwissLipids' : 26}
		self.base_url = ['https://metaspace2020.eu/annotations?' + 'db_id=', '&prj=', '&ds=', '&mz=']		#TODO still needed?

	def get_urls(self):
		""" return a dict including the urls to acces the datasets via the browser interface of METASPACE

		RETURNS
		-------
		urls : dict
			{dataset_name : {databse_name : url}}
		"""
		urls = {}
		if proj_id:
			for dataset_name in dataset_names:
				urls[dataset_name] = {}
				for database in self.databases:
					urls[dataset_name][database] = self.base_url[0] + database_ids[database] + self.base_url[1] + proj_id + self.base_url[2] + dataset_name
		else:
			for dataset_name in dataset_names:
				urls[dataset_name] = {}
				for database in self.databases:
					urls[dataset_name][database] = self.base_url[0] + database_ids[database] + self.base_url[2] + dataset_name
		return urls

	def get_datasets(self, drop_duplicates = False, sort_by = None):
		"""get all selected datasets from METASPACE

		PARAMETERS
		----------
		drop_duplicates : Bool
			if True, than duplicates will be dropped without following a hierarchy or averaging in remaining parameters, optional default : False
			#HERE ALL DATABASE INFORMATION IS REMOVED AND RESULTING DUPLICATES WILL BE DROPPED
			#DIFFERING INFORMATION IN DIMENSIONS SUCH AS MSM FDR INTENSITY WILL BE LOST
		sort_by_mz : list
			list of columns to sort by, optional
		"""
		self.data = pd.DataFrame()
		for dataset_name in self.dataset_names:
			for database in self.databases:
				dataset = self.get_dataset(dataset_name, self.database_ids[database])
				self.data = pd.concat([self.data, dataset])		#concatenate the dataset to one big dataframe containing all datasets
		if drop_duplicates:
			self.data.drop_duplicates(subset = ['mz', 'ionFormula', 'ion', 'dataset'], inplace = True)		#remove resulting duplicates
		if sort_by is not None:
			self.data.sort_values(sort_by, inplace = True)

	def get_dataset(self, dataset_name, database_id):
		"""get a single dataset from METASPACE

		PARAMETERS
		----------
		dataset_name : str
			name of the dataset (e.g. timestamp)
		database_id : int
			integer identifier of the database

		RETURNS
		-------
		dataset : DataFrame
			DataFrame containing the annotation information
		"""
		dataset = self.sm.dataset(id = dataset_name).results(database = database_id, fdr = self.max_fdr)		#load the dataset
		dataset['databases'] = database_id		#add a databases tag
		dataset['dataset'] = dataset_name		#add a dataset tag
		dataset['n_annotations'] = dataset['mz'].values.shape[0]		#add the number of found annotations on the dataset and the databases
		dataset['formula'] = dataset.index.get_level_values(0)		#add a formula tag, generated from the index information
		dataset['adduct'] = dataset.index.get_level_values(1)		#add a adduct tag, generated from the index information

		def _get_uniquelipidgroup(names):
			"""split the moleculenames for grouping

			PARAMETERS
			----------
				names : list
				list of the molecule names

			RETURNS
			-------
				np.unique(groups) : lis
				the unique remaining names after splitting
			"""
			groups = []
			for name in names:
				groups.append(name.split('(')[0])
			return np.unique(groups)

		dataset['moleculeGroups'] = [_get_uniquelipidgroup(names) for names in dataset['moleculeNames']]
		return dataset

	def filter_by_group(self, mygroups):
		"""filter the dataset by matching the molecule group in the moleculeNames

		PARAMETERS
		----------
			mygroups : list
			list of meleculegroup indentifiers
		"""
		def filtergroups(mygroups, groups):
			value = []
			for mygroup in mygroups:
				for group in groups:
					value.append(mygroup in group)
			return any(value)

		boolarr = [filtergroups(mygroups, groups) for groups in self.data['moleculeNames']]
		self.data = self.data[boolarr]

	def filter_by(self, parameters = ['msm'], values = [.8], operators = ['>']):
		"""filter the dataset by comparing the given set of parameters and values using the operators

		PARAMETERS
		----------
		parameters : [str]
			list of parameter identifiers
		values : list
			list of filter-values
		operators : [str]
			list of operators for evaluation
		"""
		for parameter, value, operator in zip(parameters, values, operators):
			self.data.query(parameter + operator +  str(value), inplace = True)

	def filter_neighboring_mzs(self, rrange = 10e-6):
		"""remove the second of two neighboring annotations if mz value is inside the range - works only if sorted by mz

		PARAMETERS
		----------
		rrange : float
			relative range in which values are removed
		"""
		oldmz = 0
		for index, row in self.data.iterrows():
			if (row['mz']-oldmz) < rrange*row['mz']:
				self.data.drop(index, inplace=True)
			else:
				oldmz = row['mz']

	def get_image(self, dataset_name, database, mz):
		"""get a dataset image to adduct

		PARAMETERS
		----------
		dataset_name : str
			name of the dataset (e.g. timestamp)
		database : str
			string identifier of the database
		mz : float
			the mz of the isotope

		RETURNS
		-------
		img : array, shape = [rows, columns]
			2D-array containing the pixel intensities mz
		"""
		if dataset_name in self.data.dataset.values:
			if np.array([self.data.mz == mz]).any():
				formula = self.data.formula[self.data.mz == mz]
				adduct = self.data.adduct[self.data.mz == mz]
				img = self.sm.dataset(id = dataset_name).isotope_images(sf = formula.values[0], adduct = adduct.values[0], only_first_isotope=True)[0]
				return img
			else:
				raise ValueError('mz not annoatated')
		else:
			dataset = self.get_dataset(dataset_name, self.database_ids[database])
			if np.array([dataset.mz == mz]).any():
				formula = dataset.formula[dataset.mz == mz]
				adduct = dataset.adduct[dataset.mz == mz]
				img = self.sm.dataset(id = dataset_name).isotope_images(sf = formula.values[0], adduct = adduct.values[0], only_first_isotope=True)[0]
				return img
			else:
				raise ValueError('mz not annoatated')

