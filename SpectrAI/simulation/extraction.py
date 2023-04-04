import numpy as np
from scipy.interpolate import interp1d

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# ------------------------------------------------------
# Clean the spectrum according to the light source range
def _clean_light_range(
	wv: np.ndarray, coeff: np.ndarray,
	light_min: float, light_max: float,
	padding: float = 10
	):

	"""_clean_light_range function
---------------------------
Get the extinction coefficient of the sample using interpolation

Input(s):
- wv, coeff {np.array}: Data to be interpolated, respectively the wavelength and the corresponding
					extinction coefficient of the molecule.
- light_min, light_max {float}: Limit of the range of the source light. Everything
								outside this range will be made equal to 0.
- padding {float}: (Opt.) Width of the transition between the range [light_min; light_max] and the rest
				   of the spectrum (A=0).
				   Default: 10 nm.

Output(s):
- cleaned_coeff {np.array}: New graph obtained after cleaning.
"""

	# Initialize the new array
	clean_matrix = np.zeros(shape=coeff.shape)

	# Set the central part to 1
	clean_matrix[(wv >= light_min+padding)&(wv <= light_max-padding)] = 1.

	# Clean the edges of the signal
	base_mask = (wv <= light_min) | ((wv >= light_min+padding)&(wv <= light_max-padding)) | (wv >= light_max)
	base_wv = wv[base_mask]
	clean_matrix = clean_matrix[base_mask]

	# Interpolate the missing edges with lines
	f_inter = interp1d(base_wv, clean_matrix, kind='linear')
	clean_matrix = f_inter(wv)

	# Apply the filter to the value
	cleaned_coeff = coeff * clean_matrix

	return cleaned_coeff

# ------------------------
# Clean the given spectrum
def _clean_spectrum(
	wv: np.ndarray, coeff: np.ndarray,
	wv_min: float, wv_max: float,
	min_NIR: float
	):

	"""_clean_spectrum function
------------------------
Get the extinction coefficient of the sample using interpolation

Input(s):
- wv, coeff {np.array}: Data to be interpolated, respectively the wavelength and the corresponding
					extinction coefficient of the molecule.
- wv_min, wv_max {float}: Min and max wavelength to use to preload the interpolation
- min_NIR {float}: Limit of the NIR detection. Delete all signal below that value

Output(s):
- wv, coeff {np.array}: New graph obtained after interpolation.
"""

	# Clean the data from NIR
	nir_mask = (wv >= min_NIR)
	wv = wv[nir_mask]
	coeff = coeff[nir_mask]

	# Complete data an initial point
	wv = np.insert(wv, 0, wv_min)
	coeff = np.insert(coeff, 0, 0)

	# Complete data with a final point if missing
	if np.amax(wv) < wv_max:
		wv = np.append(wv, wv_max)
		coeff = np.append(coeff, 0)

	return wv, coeff

# --------------------------------------
# Make the actual complete interpolation
def _complete_interpolation(
	coeff: np.ndarray,
	wv_ref: np.ndarray,
	wv_min: float, wv_max: float,
	kind: str = 'cubic',
	min_limit: float = 0,
	wv_step: int = 1
	):

	"""_complete_interpolation function
--------------------------------
Get the extinction coefficient of the sample using interpolation

Input(s):
- coeff {np.array}: Extinction coefficient of the molecule to be interpolated
- wv_ref {np.array}: Output wavelength list to use for the interpolation.
- wv_min, wv_max {float}: Min and max wavelength to use to preload the interpolation
- kind {str}: (Opt.) Type of function to use for the interpolation.
			  Default: cubic
- min_limit {int}: (Opt.) wavelength value to use to preload the interpolation.
				   Default: 0 (take all the wavelength)
- wv_step {int}: Step in wavelength to use for the interpolation.
				 Default: 100 (take only 1 value every 100)

Output(s):
- inter_coeff {np.array}: New graph obtained after interpolation.
"""

	# Select the interpolation range
	preint_coeff = coeff[wv_ref>min_limit][::wv_step]
	preint_wv = wv_ref[wv_ref>min_limit][::wv_step]

	# Add points to make sure the interpolation range is wide enough
	preint_wv = np.insert(preint_wv, 0, wv_min) # First point
	preint_coeff = np.insert(preint_coeff, 0, 0)
	preint_wv = np.append(preint_wv, wv_max) # Last point
	preint_coeff = np.append(preint_coeff, 0)

	# Run the interpolation
	f_inter = interp1d(preint_wv, preint_coeff, kind=kind)
	inter_coeff = f_inter(wv_ref)

	return inter_coeff

# --------------------------------------------------------------------------
# Complete the missing point in a graph with a sliding average interpolation
def _sliding_interpolation(
	wv: np.ndarray, coeff: np.ndarray,
	wv_ref: np.ndarray,
	window: int = 2
	):

	"""_sliding_interpolation function
-------------------------------
Interpolate the signal using a sliding average filter

Input(s):
- wv, coeff {np.array}: Data to be interpolated, respectively the wavelength and the corresponding
					extinction coefficient of the molecule.
- wv_ref {np.array}: Output wavelength list to use for the interpolation.
- window {int}: (Opt.) Window size to use for the sliding average
				Default: 2

Output(s):
- new_coeff {np.array}: Filtered data
"""
	
	# Initialise the list
	new_coeff = []

	# Process all the data in the reference
	for crt_wv in wv_ref:

		# Calculate the "distance to the point"
		d = abs(wv - crt_wv)
		d_id = d.argsort()

		# Calculate the average absorbance
		new_coeff.append( np.mean(coeff[d_id[:window]]) )

	# Convert to an array
	new_coeff = np.array(new_coeff)

	return new_coeff

##-\-\-\-\-\-\-\-\
## PUBLIC FUNCTIONS
##-/-/-/-/-/-/-/-/

# --------------------------------------------------
# Interpolate data on the reference wavelength range
def getCoefficient(
	wv: np.ndarray, coeff: np.ndarray,
	wv_ref: np.ndarray,
	kind: int = 'cubic',
	light_min: float = 430, light_max: float = 800,
	padding: float = 10,
	min_NIR: float = None,
	window: int = 2,
	min_limit: float = 0,
	wv_step: int = 100
	):

	"""getCoefficient function
-----------------------
Get the extinction coefficient of the sample using interpolation

Input(s):
- wv, coeff {np.array}: Data to be interpolated, respectively the wavelength and the corresponding
					extinction coefficient of the molecule.
- wv_ref {np.array}: Output wavelength list to use for the interpolation.
- kind {str}: (Opt.) Type of function to use for the interpolation.
			  Default: cubfric
- light_min, light_max {float}: (Opt.) Limit of the range of the source light. Everything
								outside this range will be made equal to 0.
								Default: min = 430 nm, max = 800 nm (Thorlabs LED light source).
- padding {float}: (Opt.) Width of the transition between the range [light_min; light_max] and the rest
				   of the spectrum (A=0).
				   Default: 10 nm.
- min_NIR {float}: (Opt.) Limit of the NIR detection. Delete all signal below that value
				   Default: None (calculate from the wv_ref array)
- window {int}: (Opt.) Window size to use for the sliding average interpolation
				Default: 2.
- min_limit {int}: (Opt.) wavelength value to use to preload the interpolation.
				   Default: 0 (take all the wavelength)
- wv_step {int}: Step in wavelength to use for the interpolation.
				 Default: 100 (take only 1 value every 100)

Output(s):
- wv_ref, coeff_processed {np.array}: New graph obtained after interpolation.
"""

	# Get the limits of the reference values
	wv_min = np.amin(wv_ref) - 1
	wv_max = np.amax(wv_ref) + 1

	# Clean the spectrum
	if min_NIR is None:
		min_NIR = np.amin(wv_ref)
	wv, coeff = _clean_spectrum(wv, coeff, wv_min, wv_max, min_NIR)

	# Perform a quick and dirty sliding interpolation
	coeff_inter = _sliding_interpolation(wv, coeff, wv_ref, window=2)

	# Remove the eventual signal outside the light source range
	coeff_inrange = _clean_light_range(wv_ref, coeff_inter, light_min, light_max, padding=padding)

	# Complete the interpolation
	coeff_processed = _complete_interpolation(coeff_inrange, wv_ref, wv_min, wv_max, kind=kind, min_limit = min_limit, wv_step = wv_step)

	return wv_ref, coeff_processed