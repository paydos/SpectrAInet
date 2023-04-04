import math
import numpy as np

##-\-\-\-\-\-\-\
## MATH FUNCTIONS
##-/-/-/-/-/-/-/

# --------------------------
# Sigmoid Boltzmann function
def _sigmoid(x: float, a1: float, a2: float, x0: float, dx: float):

    """_sigmoid function
-----------------
Function defining a sigmoid Boltzmann function
More information: https://en.wikipedia.org/wiki/Sigmoid_function

Input(s):
- x {float or np.ndarray}: Input value
- a1 {float}: First plateau of the sigmoid
- a2 {float}: Second plateau of the sigmoid
- x0 {float}: Central value of the sigmoid
- dx {float}: Rate of the transition range of the sigmoid
    
Output(s):
- {float or np.ndarray}: Output value
"""

    return (a1 - a2) / (1 + np.exp((x-x0)/dx)) + a2

# ----------------------------------------------------------
# Calculate the slope of the transition range of the sigmoid
def _get_sigmoid_slope(a1: float, a2: float, dx: float):

    """_get_sigmoid_slope function
---------------------------
Get the slope of the transition range of the sigmoid from its rate

Input(s):
- a1 {float}: First plateau of the sigmoid
- a2 {float}: Second plateau of the sigmoid
- dx {float}: Rate of the transition range of the sigmoid
    
Output(s):
- {float}: Slope of the transition range of the sigmoid
"""

    return (a2 - a1) / (4 * dx)

# ---------------------------------------------------
# Get the delay from the central point of the sigmoid
def _get_sigmoid_delay(a1: float, a2: float, dx: float):

    """_get_sigmoid_delay function
---------------------------
Calculate the delay between the central point of the sigmoid
and the beginning of the transition rate - assuming x0 = 0.

Input(s):
- a1 {float}: First plateau of the sigmoid
- a2 {float}: Second plateau of the sigmoid
- dx {float}: Rate of the transition range of the sigmoid
    
Output(s):
- {float}: Time at which the transition rate of the sigmoid starts
"""

    # Get the slope
    a = _get_sigmoid_slope(a1, a2, dx)
    b = (a1 + a2) / 2

    # Get the two transition positions
    t0 = (a1 - b) / a

    return t0

##-\-\-\-\-\-\-\-\-\
## PRIVATE FUNCTIONS
##-/-/-/-/-/-/-/-/-/

# ---------------------------------------------------
# Convert a list of times into a sequence of sigmoids
def _make_sigmoid_signal(
    x: np.ndarray,
    ts: list,
    c0: float, c1: float,
    dx: float
    ):

    """_make_sigmoid_signal function
-----------------------------
Convert an input list of times into a sequence of alternating
positive and negative sigmoids, assuming the times are the central
points of the different sigmoids.
In this model, all sigmoids share the same plateaux (c0 and c1) and
the same transition rate (dx).

Input(s):
- x {np.ndarray}: Input array to use for the data generation
- ts {list of float}: List of all the time / central points of the sigmoids
- c0, c1 {float}: Low and high plateaux
- dx {float}: Rate of the transition range of the sigmoid
    
Output(s):
- y {float}: Output array
"""
  
    # Initialise the signal array
    y = np.zeros(shape=x.shape)

    # Process all the times
    for i, t in enumerate(ts):

        # Get the high and low level
        a1, a2 = c0, c1
        if i%2:
            a1, a2 = c1, c0

        # Add the sigmoid to the signal
        y += _sigmoid(x, a1, a2, t, dx)

    # Correct the baseline
    y -= math.floor(len(ts)/2) * (c1 + c0)

    return y

# ----------------------------------
# Convert the input string into bits
def _str_to_bits(
    s: str
    ):

    """_str_to_bits function
---------------------
Convert the input string into a byte (8 bits)
Code taken from https://stackoverflow.com/questions/10237926/convert-string-to-list-of-bits-and-viceversa

Input(s):
- s {str}: Input string to convert to bits
    
Output(s):
- result {list of int}: List of all the bits corresponding to the input text
"""

    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

##-\-\-\-\-\-\-\-\
## PUBLIC FUNCTIONS
##-/-/-/-/-/-/-/-/

# --------------------------------------------------------------------------
# Generate the concentration over time evolution from the input text message
def generateMessage(
    msg: str,
    bit_time: float,
    duty_cycle: float = .6,
    dt: int = 5,
    t0: float = 0,
    c0: float = 0., c1: float = 1.,
    dx: float = 36
    ):

    """generateMessage function
------------------------
Convert an input text message into a simulated evolution of the concentration
over time.

Input(s):
- msg {str}: Input message to encode into the evolution of concentration
- bit_time {float}: Bit interval to use for the message encoding
                    The bit interval has to be expressed in minutes.
- duty_cycle {float}: (Opt.) Duty cycle to use for the bit-1 generation.
                      Default: 0.6 (from experimental test)
- dt {int}: (Opt.) Sampling rate of the detector emulated for this test.
            The sampling rate has to be expressed in seconds.
            Default: 5 (from experimental test)
- t0 {float}: (Opt.) Time at which the message transmission starts. Can be used
              to emulate delays in the transmission between two messages.
              Default: 0 (no delay)
- c0, c1 {float}: (Opt.) The low and high concentrations used to encode the message using BCSK.
                  Default: respectively 0 (no concentration) and 1 (normalized concentration)
- dx {float}: (Opt.) Rate of the transition range of the sigmoid
              Default: 36 (measured experimentally)
    
Output(s):
- x, y {np.ndarray}: Time and concentration array corresponding to the message
"""

    # Convert to seconds
    bit_time *= 60

    # Get the pump reaction delay
    pump_delay = abs( _get_sigmoid_delay(c0, c1, dx) )

    # Get the message into bits
    bit_sequence = _str_to_bits(msg)

    # Add two bit-1 at the beginning as well as a wait time, and a wait time at the end
    processed_bit_sequence = [0, 0, 1,1] + bit_sequence + [0,0,0,0]

    # Process all the bits
    all_times = []
    last_t = t0
    for bit in processed_bit_sequence:

        # React if bit-1 - do nothing if bit-0
        if bit:

            # Add the increasing edge
            all_times.append(last_t + pump_delay)

            # Add the decreasing edge
            all_times.append(last_t + pump_delay + (bit_time * duty_cycle))
      
        # Move to the next bit
        last_t += bit_time

    # Generate the signal
    x = np.arange(0, last_t, dt)
    y = _make_sigmoid_signal(x, all_times, c0, c1, dx)

    return x, y

# ----------------------
# Merge several messages
def mergeMessages(
    ts: list, cs: list
    ):

    """mergeMessages function
----------------------
Merge different messages encoded into concentration evolution of different
length into a single array.

Input(s):
- ts, cs {list of np.ndarray}: list of all the time and concentration arrays for each message
    
Output(s):
- time_array, concentration_array {np.ndarray}: Merged array
"""
    
    # Find the longest message in the list
    len_list = [len(x) for x in ts]
    max_id = np.argmax(len_list)
    max_len = np.amax(len_list)

    # Get the common time array
    time_array = ts[max_id]

    # Pad all the concentration arrays for the max length
    new_cs = []
    for c in cs:

        if len(c) != max_len:
            n_pad = max_len - len(c)
            new_c = np.pad(c, (0, n_pad), 'constant')
        else:
            new_c = np.copy(c)

        new_cs.append(new_c)

    # Stack all the arrays together
    concentration_array = np.dstack(new_cs)[0]

    return time_array, concentration_array