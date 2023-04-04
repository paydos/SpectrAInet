from SpectrAI.simulation.extraction import getCoefficient
from SpectrAI.simulation.generation import concentrationScaling, createMixture, generateSamples
from SpectrAI.simulation.input_output import readFile, readFolder, saveSamples
from SpectrAI.simulation.message import generateMessage, mergeMessages
from SpectrAI.simulation.single_generation import generateINDVsamples

##-\-\-\-\-\-\-\-\-\-\-\-\-\-\
## MESSAGE GENERATION FUNCTIONS
##-/-/-/-/-/-/-/-/-/-/-/-/-/-/

# ---------------------------
# Generate a group of message
def makeGroupMessage(
    messages: list,
    bit_time: float,
    c0: float = 0., c1: float = 1.,
    duty_cycle: float = 0.6, 
    dt: int = 5,
    t0: float = 0,
    dx: float = 36
    ):

    """makeGroupMessage function
-------------------------
Genereate several messages, all encoded into different simultaneous concentration evolutions.

Input(s):
- messages {list of str}: list of all the test messages to send simultaneously
- bit_time {float or list of float}: bit interval to use to encode the messages. If a single
                                     float is provided, all messages will use the same bit
                                     interval.
- c0, c1 {float or list of float}: (Opt.) The low and high concentrations used to encode the
                                   messages using BCSK. If single floats are provided, all
                                   messages will use the same concentrations.
                                   Default: respectively 0 (no concentration) and 1
                                   (normalized concentration)
- duty_cycle {float or list of float}: (Opt.) Duty cycle to use for the bit-1 generation.
                                       If a single float is provided, all messages will use
                                       the same duty cycle.
                                       Default: 0.6 (from experimental test)
- dt {int}: (Opt.) Sampling rate of the detector emulated for this test.
            The sampling rate has to be expressed in seconds.
            Default: 5 (from experimental test)
- t0 {float}: (Opt.) Time at which the message transmission starts. Can be used to emulate
              delays in the transmission between two messages. If a single float is provided,
              all messages will use the same delay.
              Default: 0 (no delay)
- dx {float}: (Opt.) Rate of the transition range of the sigmoid
              Default: 36 (measured experimentally)
    
Output(s):
- time, concentrations {np.ndarray}: Array containing the evolution of the signals over time.
"""

    # Get the number of messages to encode
    n_msg = len(messages)

    # Convert all provided floats into list if needed
    if not isinstance(bit_time, list):
        bit_time = [bit_time] * n_msg
    if not isinstance(duty_cycle, list):
        duty_cycle = [duty_cycle] * n_msg
    if not isinstance(c0, list):
        c0 = [c0] * n_msg
    if not isinstance(c1, list):
        c1 = [c1] * n_msg
    if not isinstance(t0, list):
        t0 = [t0] * n_msg

    # Generate all the individual messages
    all_times = []
    all_concentrations = []

    for i in range(n_msg):

        # Make the message
        crt_t, crt_c = generateMessage(messages[i], bit_time[i], duty_cycle=duty_cycle[i], c0=c0[i], c1=c1[i], dt=dt, t0=t0[i], dx=dx)

        # Append to structures
        all_times.append(crt_t)
        all_concentrations.append(crt_c)

    # Merge all the messages in the same array
    time, concentrations = mergeMessages(all_times, all_concentrations)

    return time, concentrations