import pandas as pd
def get_adducts(masses = None, formulas = None, adducts = ['M+H', 'M+Na', 'M+K']):
	"""return possible adducts and their respective masses to either a set of masses or molecular formulas
		This function is based on the MSAC package https://github.com/pnnl/MSAC

		PARAMETERS
		----------
		masses : list of float
			masses in atomic mass units, either masses or formulas must be provided. If both, masses will be used
		formulas : list of str
			molecular formulas, either masses or formulas must be provided. If both, masses will be used
		adducts : list of str, default = ['M+H', 'M+Na', 'M+K']
			adducts to return if applicable. See MSAC documentation for all considered adducts

		RETURNS
		------
		mols_adducts : pd.DataFrame
			with columns of mass, adduct, adduct mass and optionally formula of the selected adducts
	"""
	from msac import process
	if ((masses is not None) & (formulas is not None)):
		import warnings
		warnings.warn('Ambiguous keyword arguments masses AND formulas given. Using masses, ignoring formulas')
	if masses is not None:
		mols_adducts = process.process_file(pd.DataFrame({'mass' : masses}), mass_col = 'mass')
	elif formulas is not None:
		mols_adducts = process.process_file(pd.DataFrame({'formula' : formulas}), mass_col = 'mass', no_mass_formula_col = 'formula')
	else:
		raise TypeError('Missing masses or formulas keyword argument')
	return mols_adducts[mols_adducts['adduct'].isin(adducts)]

def get_isotopes(formulas, molecular_abundance_threshold = 1e-2, isotope_abundance_threshold = 5e-4):
	"""calculate isotope combinations of molecular formulas filtered by molecular and single atomic isotope abundance
		This function is based on the pyteomics package https://github.com/levitsky/pyteomics

	PARAMETERS
	----------
	formulas : list of str
		molecular formulas
	molecular_abundance_threshold : float, default 1e-2
		molecular abundance threshold to include isotopic molecular composition in the returned compositions
	isotope_abundance_threshold : float, default = 5e-4
		isotope abundance threshold to include isotope in molecular composition

	RETURNS
	-------
	formula_dict : dict of dict of tuple, shape = [3]
		dictionary including the provided formulas and their isotope compositions meeting the abundance threshold, their mass and their absolute and relative abundance
		the following pattern is used:
		{formula : {isotopic composition : (mass, absolute abundance, relative abundance)}}
	"""
	from pyteomics import mass as molmass
	formula_dict = dict(zip(formulas, [{} for _ in range(len(formulas))]))
	for mol in formulas:
		first_abundance = None
		for isotope, abundance in molmass.isotopologues(mol, report_abundance = True, isotope_threshold = isotope_abundance_threshold):
			if abundance > molecular_abundance_threshold:
				if first_abundance is None:
					first_abundance = abundance
				formula = ''.join([isotope_string + str(amount) for isotope_string, amount in isotope.items()])
				mass = molmass.calculate_mass(isotope)
				formula_dict[mol][formula] = (mass, abundance, abundance/first_abundance)
	return formula_dict