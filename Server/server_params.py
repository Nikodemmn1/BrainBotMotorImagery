from SharedParameters.signal_parameters import BIOSEMI_CHANNELS_COUNT


CHANNELS = BIOSEMI_CHANNELS_COUNT # field "Channels sent by TCP" in Actiview
SAMPLES = 128  # field "TCP samples/channel" in Actiview
WORDS = CHANNELS * SAMPLES

DECIMATION_FACTOR = 30

SERVER_BUFFER_LEN = 3200
SERVER_OVERLAP = 128

SAMPLES_DECIMATED = SAMPLES // DECIMATION_FACTOR

TCP_LOCAL_PORT = 7230

# TCP_AV_PORT = 8188  # port configured in Activeview
TCP_AV_PORT = 8888  # port configured in Activeview
TCP_AV_ADDRESS = 'localhost'  # IP adress of Actiview host

UDP_PORT = 5502
UDP_IP_ADDRESS = 'localhost'

REMOTE_UDP_PORT = 5500
REMOTE_UDP_ADDRESS = "localhost"

MEAN_PERIOD_LEN = 8192
