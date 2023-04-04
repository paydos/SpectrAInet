from glob import glob
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tqdm import tqdm

##-\-\-\-\-\-\-\-\-\-\-\
## PUBLIC INPUT FUNCTIONS
##-/-/-/-/-/-/-/-/-/-/-/

# --------------------------------
# Load the CNN model from the file
def loadModel(
	fpath: str
	):

	"""loadModel function
------------------
Function to load the training set from the files.
    
Input(s):
- fpath {str}: Name of the file containing the model. Can be either a HDF5 file
			   (raw model) or a pickle file (model + history + molecule map)

Output(s):
- model {keras Sequential}: Trained CNN model to be saved
- molecule_map {dict}: Dictionary of the label used for the training
- history {keras history}: Information on the loss and metrics used for the training
"""

	# Directly load from HDF5 format
	if os.path.splitext(fpath)[-1] == '.h5':
		model = tf.keras.models.load_model(fpath)
		molecule_map, history = None, None
	
	# Load from a pickle file
	else:
		model, molecule_map, history = pickle.load(open(fpath, 'rb'))

	return model, molecule_map, history

# ------------------------------------
# Load the training dataset from files
def loadTrainingSet(
	dataset_path: str,
	label_path: str = None
	):

	"""
loadTrainingSet function
------------------------
Function to load the training set from the files.
    
Input(s):
- dataset_path {str}: Name of the file containing the absorbance (x) of the training set.
					  The file can be either a pickle (.pkl) file, or a text file (.csv).
					  In the case of .csv file, a second file should be provided using label_path=
					  for the label and molecule map.
- label_path (str): (Opt.) Name of the file containing the label and molecule map, in case
					of using text file (.csv).
					Default: None
"""

	# Open a pickle file
	if label_path is None:
		x, y, molecule_map = pickle.load(open(dataset_path, 'rb'))

	# Open a collection of csv files
	else:

		# Get the absorbance
		x = np.loadtxt(dataset_path, delimiter=',')

		# Get the concentration
		y = np.loadtxt(label_path, delimiter=',', skiprows=1)

		# Get the molecule map
		with open(label_path, 'r') as label_file:
			map_line = label_file.readline()

		# Clean the molecule map
		if map_line[0] == '#':
			map_line = map_line[1:]

		molecule_map = dict((molecule.strip(), i) for i, molecule in enumerate(map_line.split(',')))

	return x, y, molecule_map

##-\-\-\-\-\-\-\-\-\-\-\-\
## PUBLIC OUTPUT FUNCTIONS
##-/-/-/-/-/-/-/-/-/-/-/-/

# -----------------------------------------------------
# Save the model and the relevant information in a file
def saveModel(
	file_name: str,
	model,
	history: dict = None,
	molecule_map: dict = None,
	extension: str = None,
	):

	"""saveModel function
------------------
Function to save the trained model, along with additional informations, in a file.
    
Input(s):
- file_name {str}: Base name to use to save the file
- model {keras Sequential}: Trained CNN model to be saved
- history {dict}: (Opt.) Analysis of the training of the CNN and its efficiency.
				  Default: None
- molecule_map {dict}: (Opt.) Map of the molecules used to generate the training set.
					   Default: None
- extension {str}: (Opt.) Extension to use to save the data in files.
				   Default: None (automatically guess the file extension)
"""
	
	# Check the data to save
	keras_only = False
	if history is None and molecule_map is None:
		keras_only = True

	# Get the extension
	if extension is None:
		if keras_only:
			extension = '.h5'
		else:
			extension = '.pkl'

	# Save the data
	if keras_only:
		tf.keras.models.save_model(model, file_name+extension)
	else:
		pickle.dump([model, molecule_map, history], open(file_name+extension, 'wb'))

# -----------------------------
# Save the training set to file
def saveTrainingSet(
	base_name: str,
	absorbance: np.ndarray,
	concentration: np.ndarray,
	molecule_map: dict,
	extension: str = None,
	use_pickle: bool = False
	):

	"""saveTrainingSet function
------------------------
Function to save all the training set inside a pair of files
    
Input(s):
- base_name {str}: Name to use as a base to create the name of the files
- samples {np.ndarray}: Array containing the absorbance of all the available samples.
- concentration {np.ndarray}: Array containing the concentration used to generate all the samples.
- molecule_map {dict}: Map of the molecules used to generate the training set.
- extension {str}: (Opt.) Extension to use to save the data in files.
				   Default: None (automatically guess the file extension)
- use_pickle {bool}: (Opt.) Save in a pickle file instead of a collection of csv files.
"""

	# Get the extension
	if extension is None:
		if use_pickle:
			extension = '.pkl'
		else:
			extension = '.csv'

	# Save the data using pickles
	if use_pickle:
		
		# Make the file name
		fpath = base_name + extension

		# Save
		pickle.dump([absorbance, concentration, molecule_map], open(fpath, 'wb'))

	# Save the data as text files
	else:

		# Make the file names
		absorbance_path = base_name + '_x' + extension
		concentration_path = base_name + '_y' + extension

		# Make the header from the molecule map
		header_list = [''] * len(molecule_map.keys())
		for mol_name, mol_id in molecule_map.items():
			header_list[mol_id] = mol_name

		header = header_list[0]
		for name in header_list[1:]:
			header += ',' + name

		# Save the data
		np.savetxt(absorbance_path, absorbance, delimiter=',')
		np.savetxt(concentration_path, concentration, header=header, delimiter=',')
