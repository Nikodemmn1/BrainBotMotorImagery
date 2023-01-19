BUFFER_LENGTH = 800
OVERLAP = 80

#BIOSEMI_CHANNELS_COUNT = 17 - WITH TRIGGERS
BIOSEMI_CHANNELS_COUNT = 17

# EEG signal ranges and unit
DIGITAL_MIN = -8388608.0
DIGITAL_MAX = 8388607.0
PHYSICAL_MIN = -262144.0
PHYSICAL_MAX = 262143.0
UNIT = 1e-6  # μV

CAL = (PHYSICAL_MAX - PHYSICAL_MIN) / (DIGITAL_MAX - DIGITAL_MIN)
OFFSET = PHYSICAL_MIN - DIGITAL_MIN * CAL

DATASET_FREQ = 2048

LOW_PASS_FREQ_PB = 30
LOW_PASS_FREQ_SB = 60
HIGH_PASS_FREQ_PB = 6
HIGH_PASS_FREQ_SB = 3

WELCH_OVERLAP_PERCENT = 80
WELCH_SEGMENT_LEN = 350

MAX_LOSS_PB = 2
MIN_ATT_SB = 6
