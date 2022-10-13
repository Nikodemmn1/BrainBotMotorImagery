from SharedParameters.signal_parameters import BIOSEMI_CHANNELS_COUNT, BIOSEMI_FREQ, DATASET_FREQ, BUFFER_LENGTH, \
    OVERLAP

CHANNELS = BIOSEMI_CHANNELS_COUNT + 1  # field "Channels sent by TCP" in Actiview
SAMPLES = 128  # field "TCP samples/channel" in Actiview
WORDS = CHANNELS * SAMPLES

DECIMATION_FACTOR = BIOSEMI_FREQ / DATASET_FREQ

SERVER_BUFFER_LEN = int(BUFFER_LENGTH * DECIMATION_FACTOR)
SERVER_OVERLAP = int(OVERLAP * DECIMATION_FACTOR)

SAMPLES_DECIMATED = SAMPLES // DECIMATION_FACTOR

TCP_LOCAL_PORT = 7279

TCP_AV_PORT = 8888 # port configured in Activeview
TCP_AV_ADDRESS = 'localhost'  # IP adress of Actiview host

UDP_PORT = 5502
UDP_IP_ADDRESS = 'localhost'

REMOTE_UDP_PORT = 5500
REMOTE_UDP_ADDRESS = "localhost"