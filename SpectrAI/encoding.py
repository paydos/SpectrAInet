from glob import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# ------------------------------
# Check if the string is a float
def _is_float(
    input: str
    ):

    """_is_float function
------------------
Check if the input string can be a float.
    
Input(s):
- input {str}: String to check.
    
Output(s):
- is_float {bool} : Return True is the string can be a float, False otherwise
--------
"""

    try:
        float(input)
        return True
    except ValueError:
        return False

# ------------------------------------------------------------------------
# Get the list of molecules and concentrations from the header of the file
def _molecule_list_from_fname(
    file_path: str
    ):

    """_molecule_list_from_fname function
----------------------------------
Get the list of molecules from the header of the given file
    
Input(s):
- file_path {str}: File to read
    
Output(s):
- molecule_list {list of str}: List of molecules names found in the file name
- molecule_concentrations {list of float}: List of the concentrations of the molecules
- file_id {int}: ID of the file, if found in the name
--------
"""
    
    # Extract the file name
    fname = os.path.splitext( os.path.split(file_path)[-1] )[0]

    # Split the elements from the name
    molecule_list = fname.split('_')

    # Remove the first element in the list if number - shall be fl
    file_id = None
    if len(molecule_list)%2 != 0:
        file_id = int( molecule_list[-1] )
        molecule_list = molecule_list[:-1]

    # Get all the names and concentrations
    molecule_names = []
    molecule_concentrations = []
    for i in range( int(len(molecule_list)/2) ):
        molecule_names.append( molecule_list[2*i] )
        molecule_concentrations.append( float(molecule_list[2*i+1]) )

    return molecule_names, molecule_concentrations, file_id

# -----------------------------------------------------
# Get the list of molecules from the header of the file
def _molecule_list_from_file(
    file_path: str
    ):

    """_molecule_list_from_file function
---------------------------------
Get the list of molecules from the header of the given file
    
Input(s):
- file_path {str}: File to read
    
Output(s):
- molecule_list {list of str}: List of molecules names found in the file
--------
"""

    # Get the first line from the file
    with open(file_path, 'r') as cfile:
        header = cfile.readline()

    # Get the list of molecules
    molecule_list = header.split(',')[1:]
    molecule_list = [x.strip() for x in molecule_list]

    return molecule_list

##-\-\-\-\-\-\-\-\
## PUBLIC FUNCTIONS
##-/-/-/-/-/-/-/-/

# ----------------------------------------------------------------
# Create dictionary to map molecule to integer ranging from 0 to N
def mapFromFiles(
    folder_path: str,
    concentration_file: str = None,
    extension: str = '.csv',
    verbose: bool = False
    ):

    """mapFromFiles function
---------------------
Molecules stored inside the specified folder will be labeled with 0 to N.
    
Input(s):
- folder_path {str}: Folder to scan
- concentration_file {str}: (Opt.) Name of the file including all the concentrations and molecule names.
                                   Default: None (use the file names in the folder instead)
- extension {str}: (Opt.) File extension to read during the folder scan
                          Default: .csv
- verbose {bool}: (Opt.) Toggle verbose mode on or off.
                  Default: False (not verbose)
    
Output(s):
- mapping_dict {Dict}: Each molecule is associated to one number (label)
--------
"""

    # Check if concentration file is provided
    if concentration_file is not None:
        molecule_list = _molecule_list_from_file( os.path.join(folder_path, concentration_file) )

    # Extract from the files in folder
    else:

        # Get all the files in the folder
        file_list = glob( os.path.join(folder_path, '*'+extension) )

        # Scan all the files
        molecule_list = []
        for file_path in file_list:

            # Get the list of molecules
            crt_molecules, _, _ = _molecule_list_from_fname(file_path)

            # Add to the existing list
            molecule_list = molecule_list + list( set(crt_molecules) - set(molecule_list) )

    # Build the dictionary
    mapping_dict = dict((molecule, i) for i, molecule in enumerate(molecule_list))

    if verbose:
        print('Molecule mapping:')
        print(mapping_dict)

    return mapping_dict

# --------------------------------
# Encode the files from the folder
def encodeFolder(
    folder_path: str,
    molecule_map: dict = None,
    concentration_file: str = None,
    extension: str = '.csv',#
    one_hot: bool = True,
    verbose: bool = False
    ):

    """encodeOneHot function
---------------------
Process the data to output data in correct format and one-hot encode each data to its appropiate labels. 
    
Input(s):
- folder_path {str}: Folder to scan
- molecule_map {Dict}: (Opt.) Dictionary associating one id to each molecule name.
                       Default: None (generate the map here)
- concentration_file {str}: (Opt.) Name of the file including all the concentrations and molecule names.
                            Default: None (use the file names in the folder instead)
- extension {str}: (Opt.) File extension to read during the folder scan
                          Default: .csv
- one_hot {bool}: (Opt.) If True, returns presence instead of concentrations.
                  Default: True
- verbose {bool}: (Opt.) Toggle verbose mode on or off.
                  Default: False (not verbose)
    
Output(s):
- absorbance_array {np.array}: Array containing all the absorbances found in the files
- onehot_array {np.array}: Array containing all the concentrations found in the files
                           If -one_hot is True, the array is instead the presence of the molecule
- molecule_map {dict}: Dictionary used to encode the molecule
"""

    # Get the molecule map
    if molecule_map is None:
        molecule_map = mapFromFiles(folder_path, concentration_file=concentration_file, extension=extension, verbose=verbose)
    
    # Get the concentrations from a concentration file
    if concentration_file is not None:

        # Get the path
        file_path = os.path.join(folder_path, concentration_file)

        # Get the molecule list, ID and concentration
        molecule_list = _molecule_list_from_file(file_path)
        loaded_array = np.loadtxt(file_path, skiprows=1, delimiter=',')
        id_array, concentration_array = loaded_array[:,0], loaded_array[:,1:]

        # Initialise an empty array
        onehot_array = np.zeros(shape=concentration_array.shape)

        # Process all the map
        for molecule_name, new_id in molecule_map.items():
            old_id = molecule_list.index(molecule_name)

            onehot_array[:,new_id] = concentration_array[:,old_id]

        # Get the absorbances
        absorbance_array = []
        for file_id in tqdm(id_array.astype(int), desc='Reading absorption files...', disable= not verbose):

            # Load the file content
            fname = os.path.join(folder_path, str(file_id) + extension)
            fdata = pd.read_csv(fname)
        
            # Add to the datastructure
            absorbance_array.append( np.copy(fdata['A (a.u.)'].values) )

        # Convert to array
        absorbance_array = np.array(absorbance_array)

    # Get the concentrations from the file names
    else:
        
        # Get all the files
        file_list = glob( os.path.join(folder_path, '*'+extension) )

        # Initialise the arrays
        onehot_array = np.zeros( shape=(len(file_list),len(molecule_map.keys())) )
        test_data = pd.read_csv(file_list[0])
        test_array = test_data['A (a.u.)'].values
        absorbance_array = np.zeros( shape=(len(file_list),test_data.shape[0]) )

        # Process all the files
        for i, fname in enumerate(tqdm(file_list, desc='Reading absorption files...', disable= not verbose)):

            # Get the file informations
            molecule_names, molecule_concentrations, file_id = _molecule_list_from_fname(fname)

            if file_id is None:
                file_id = i

            # Input the information in the array
            for j in range(len(molecule_names)):
                col_id = molecule_map[molecule_names[j]]
                onehot_array[file_id, col_id] = molecule_concentrations[j]

            # Get the absorbance
            fdata = pd.read_csv(fname)
            absorbance_array[file_id] = np.copy(fdata['A (a.u.)'].values)

    # Encode in oneHot
    if one_hot:
        onehot_array[ onehot_array > 0 ] = 1
        onehot_array = onehot_array.astype(int)

    return absorbance_array, onehot_array, molecule_map


