import tensorflow as tf

from sklearn.model_selection import train_test_split

from fnmatch import fnmatch
import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

##-\-\-\-\-\-\-\-\
## NEURAL NETWORKS
##-/-/-/-/-/-/-/-/

# ---------------------------------------------------
# Build the CNN to detect the presence of N molecules
def _presence_CNN(
    input_shape: tuple,
    n_molecules: int,
    learning_rate: float = 0.001,
    loss: str = 'binary_crossentropy',
    metrics: list = ['binary_accuracy'],
    ):

    """_presence_CNN function
----------------------
Build the CNN-1D to detect the presence of N molecules.
    
Input(s):
- input_shape {tuple}: Shape of the array to be analyzed by the CNN
- n_molecules {int}: Number of molecules that the CNN has to identify
- loss {str}: (Opt.) Name of loss function to use to converge the training
              of the CNN.
              Default: binary_crossentropy
- metrics {list of str}: (Opt.) List of the metrics to use to evaluate the efficiency
                         of the training.
                         Default: ['binary_accuracy']
    
Output(s):
- model {keras Sequential} : CNN-1D model to be used for training
--------
"""
    
    # Build the CNN
    model = tf.keras.Sequential([
        
    # 1 Dimensional Convolutional Layers
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling1D(),
    
    # Flatten
    tf.keras.layers.Flatten(),
    
    # Dense layers    
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    
    # Final layer
    tf.keras.layers.Dense(n_molecules, activation='sigmoid')
    ])
    
    # Define the optimiser to use for the training
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the CNN
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model

##-\-\-\-\-\-\-\-\
## PUBLIC FUNCTIONS
##-/-/-/-/-/-/-/-/

# -----------------------------------------------------------
# Predict the presence of N molecules using a pre-trained CNN
def predictPresence(
    model,
    x: np.ndarray,
    threshold: float = .5
    ):

    """predictPresence function
------------------------
Build the CNN-1D to detect the presence of N molecules.
    
Input(s):
- model {keras Sequential} : CNN-1D model to use for the prediction
- x {np.ndarray}: Absorbance spectra to analyse to predict the presence of the molecules
- threshold {float}: (Opt.) Threshold to apply for the cleaning/normalization
                     Default: 0.5
    
Output(s):
- predicted_y {np.ndarray} : Predicted presence of the molecules for each of the absorbance spectra
--------
"""
    
    # Predict the presence of molecules in each x given
    predicted_y = model.predict(x)

    # Clean the result
    predicted_y[predicted_y > .5] = 1.
    predicted_y[predicted_y <= .5] = 0.

    return predicted_y

# ------------------------------------------------------
# Train the CNN-1D to detect the presence of N molecules
def trainPresence(
    x: np.ndarray, y: np.ndarray,
    # Extra arguments for the training
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    shuffle: bool = True,
    loss: str = 'binary_crossentropy',
    metrics: list = ['binary_accuracy'],
    # Extra arguments for the training set split
    split_sets: bool = True,
    test_size: float = 0.3,
    random_state: int = 40,
    # Other
    verbose: bool = True
    ):

    """trainPresence function
----------------------
Build and train the CNN-1D to detect the presence of N molecules.
    
Input(s):
- x, y {np.ndarray}: Dataset and labels to use to train the CNN
- epochs {int}: (Opt.) Number of epochs to train the model on
                Default: 50
- batch_size {int}: (Opt.) Number of samples per gradient update
                    Default: 32
- learning_rate {float}: (Opt.) Schedule for the optimizer
                         Default: 0.001
- validation_split {float}: (Opt.) Fraction of the training data to be used as validation data.
                            Default: 0.1
- shuffle {bool}: (Opt.) Whether to shuffle the training data before each epoch
                  Default: True
- loss {str}: (Opt.) Name of loss function to use to converge the training
              of the CNN.
              Default: binary_crossentropy
- metrics {list of str}: (Opt.) List of the metrics to use to evaluate the efficiency
                         of the training.
                         Default: ['binary_accuracy']
- split_sets {bool}: (Opt.) Split the input dataset into a training and a
                     validation dataset using train_test_split.
                     Default: True
- test_size {float}: (Opt.) Fraction of the input dataset to turn into a
                     validation dataset.
                     Default: 0.3 (arbitrary)
- random_state {int}: (Opt.) Seed used for the train_test_split function
                      Default: 40 (arbitrary)
- verbose {bool}: (Opt.) Toggle verbose mode on or off.
                  Default: True (not verbose)
    
Output(s):
- model {keras Sequential} : CNN-1D model to be used for training
- history {dict}
--------
"""

    # Calculate the input parameters
    input_shape = (x.shape[1], 1)
    n_molecules = y.shape[1]

    print(input_shape, n_molecules)

    # Build the CNN to train
    model = _presence_CNN(input_shape, n_molecules, learning_rate=learning_rate, loss=loss, metrics=metrics)

    # Prepare the training set
    if split_sets:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    else:
        x_train, y_train = x, y

    # Be verbose if required
    if verbose:
        print('Shape of input X: ', np.array(x_train).shape)
        print('Shape of label Y: ', np.array(y_train).shape)

        fit_verbose = 'auto'
    else:
        fit_verbose = 0

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle, verbose=fit_verbose)

    # Use the verification dataset
    if split_sets:

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

        # Save in the dictionary
        history.history['val_subset_loss'] = loss  
        history.history['val_subset_accuracy'] = accuracy

    return model, history