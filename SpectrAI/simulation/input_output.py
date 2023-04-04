from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

##-\-\-\-\-\-\-\-\-\-\-\
## PUBLIC INPUT FUNCTIONS
##-/-/-/-/-/-/-/-/-/-/-/

# ---------------------------------
# Read the content of a single file
def readFile(
	fpath: str
	):
    
    """readFile function
-----------------
Function to read the content of a single file and
extract all the relevant informations from it
    
Input(s):
- fpath {str}: Path to the file to read.
    
Output(s):
- dataframe {Pandas dataframe}: Dataframe containing all the data from the file
- molecule_name {str}: Name of the molecule being extracted
"""

    # Extract the data into a Pandas dataframe
    dataframe = pd.read_csv(fpath, index_col=False)
    
    # Rename the first column to remove eventual #
    column1_header = list(dataframe.columns.values)[0]
    if column1_header[0] == '#':
    	new_header = column1_header[1:].strip()
    
    dataframe.rename(columns={column1_header:new_header}, inplace=True)

    # Get the molecule name from the file name
    molecule_name = os.path.splitext( os.path.split(fpath)[-1] )[0]

    return dataframe, molecule_name

# -------------------------------------------
# Read and store the data found in the folder
def readFolder(
	folder: str,
	filetype: str = '.csv',
	verbose: bool = False
	):

    """readFolder function
-------------------
Function to read all the files located inside the given folder
    
Input(s):
- folder {str}: Path to the folder to scan.
- filetype {str}: (Opt.) File extension to read.
                  Default: .csv
- verbose {bool}: (Opt.) Toggle verbose mode on or off.
                  Default: False (not verbose)
    
Output(s):
- dataframe {list of pd.DataFrame}: List of the dataframes containing all the data from the folder
- molecule_name {list of str}: List of name of the molecules being extracted
"""

    # Browse the folder and get the path to all the files inside it
    files = glob( os.path.join(folder,('*' + filetype) ), recursive=True) #! Execute from main folder
    print(files)
    
    # dataframe {list}: List of all molecules stored as a pandas dataframe
    dataframe = []
    molecule_name = []

    for counter, filename in enumerate(files):

        # Extract the content of the file
        crt_data, crt_name = readFile(filename)

        # Store in lists
        dataframe.append( crt_data )    
        molecule_name.append( crt_name )    

        # Display in terminal the index assigned to each molecule
        if verbose:
            value = filename + ' stored in dataframe [' + str(counter) + ']'
            value = value.center(os.get_terminal_size().columns)
            print(value)

    if verbose:
        print('\n\n')
    
    return dataframe, molecule_name

##-\-\-\-\-\-\-\-\-\-\-\-\
## PUBLIC OUTPUT FUNCTIONS
##-/-/-/-/-/-/-/-/-/-/-/-/

# -------------------------------------------
# Save all the samples to the selected folder
def saveSamples(
	folder_path: str,
	samples: list,
	names: list = None,
	concentrations: list = None,
	extension: str = '.csv',
	iteration: bool = True,
	use_file: bool = False,
	verbose: bool = False,
	**kwargs # Additional kwargs for the data generation
	):

	"""saveSamples function
--------------------
Function to save all the samples in the given folder
    
Input(s):
- folder_path {str}: Path to the folder to save files to.
- samples {list of pd.Dataframe}: List of absorbance spectra of all the mixtures
- names {list of str}: (Opt.) List of all the molecule names found in the mixtures.
											 Used to generate the file names
    					   			 Default: None (do not use names in the file name)
- concentrations {list of float}: (Opt.) Set concentrations (excl. eventual errors)
                                  used to generate the mixtures, to add in the file names.
                                  Default: None (do not use concentrations in the file names)
- extension {str}: (Opt.) File extension to use on the files.
                   Default: .csv
- iteration {bool}: (Opt.) Use the file number in the file name.
										Forced to True if names and concentrations are None.
										Default: True
- use_file {bool}: (Opt.) Save the concentrations and names in a separate file.
				   					Avoid using names and concentrations in file names.
										Default: False
- verbose {bool}: (Opt.) Toggle verbose mode on or off.
                  Default: False (not verbose)
- **kwargs: Keyword arguments for pandas function .to_csv()
"""

	# Initialise the given folder
	os.makedirs(folder_path, exist_ok=True)

	# Check file name format
	use_molecules = not use_file

	# Save all infos in a separate file
	if use_file:

		# Get the header:
		header = 'File ID'
		for name in names[0]:
			header += ','+name

		# Convert the concentration list into an array
		concentration_arr = np.array(concentrations)

		# Add the ID
		id_arr = np.arange(0, concentration_arr.shape[0])
		data_arr = np.concatenate([id_arr[:,np.newaxis], concentration_arr], axis=1)

		# Save the file
		np.savetxt(os.path.join(folder_path,'_concentrations.csv'), data_arr, delimiter=',', header=header)

	if names is None and concentrations is None:
		iteration = True
		use_molecules = False
		if verbose:
			print('WARNING: No name or concentration given for the samples. Iteration forced for the file name generation')

	# Process all the samples
	for i, sample in enumerate(tqdm(samples, desc='Saving all samples...', disable= not verbose)):
		
		# Make the title of the file
		fname = ''

		# Add information on the molecules
		if use_molecules:

			# Get the number of molecules
			if names is None:
				n_molecules = len(concentrations[i])
			else:
				n_molecules = len(names[i])

			for j in range(n_molecules):
				if names is not None:
					fname += names[i][j] + '_'
				if concentrations is not None:
					fname += str(concentrations[i][j]) + '_'

		# Add iteration
		if iteration:
			fname += str(i)

		# Add the extension
		if fname[-1] == '_':
			fname = fname[:-1]
		fname += '.csv'

		# Save the file
		sample.to_csv( os.path.join(folder_path, fname), index=False, **kwargs)