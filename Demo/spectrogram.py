import pywt
import numpy as np
from scipy import signal, ndimage

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LogNorm, NoNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Dataset.dataset import *
from torch.utils.data import DataLoader
from SharedParameters.signal_parameters import DATASET_FREQ
from Server.server_params import DECIMATION_FACTOR
#source: https://gist.github.com/MiguelonGonzalez/00416cbf3d7f3eab204766961cf7c8fb
def cwt_spectrogram(x, fs, nNotes=12, detrend=True, normalize=True):
    N = len(x)
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x = signal.detrend(x, type='linear')
    if normalize:
        stddev = x.std()
        x = x / stddev

    ###########################################################################
    # Define some parameters of our wavelet analysis.

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2 * np.floor(N / 2.0)))
    scales = 2 ** np.arange(1, nOctaves, 1.0 / nNotes)

    #     print (scales)

    ###########################################################################
    # cwt and the frequencies used.
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef, freqs = pywt.cwt(x, scales, 'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt

    ###########################################################################
    # power
    #     power = np.abs(coef)**2
    power = np.abs(coef * np.conj(coef))

    # smooth a bit
    power = ndimage.gaussian_filter(power, sigma=2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2 * np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0 ** 2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0 / coi

    return power, times, frequencies, coif


def spectrogram_plot(z, times, frequencies, coif, cmap=None, norm=Normalize(), ax=None, colorbar=True):
    ###########################################################################
    # plot

    # set default colormap, if none specified
    if cmap is None:
        cmap = get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    xx, yy = np.meshgrid(times, frequencies)
    ZZ = z

    im = ax.pcolor(xx, yy, ZZ, norm=norm, cmap=cmap)
    ax.plot(times, coif)
    ax.fill_between(times, coif, step="mid", alpha=0.4)

    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4)
        fig.colorbar(im, cax=cbaxes, orientation='vertical')

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())

    return ax

if __name__ == "__main__":
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    labels_mapping = {'0': 'LEFT',
              '1': 'RIGHT',
              '2': 'RELAX'}
    sampling_frequency = DATASET_FREQ/DECIMATION_FACTOR
    full_dataset = EEGDataset("../DataBDF/OutNikodem/OutNikodem_train.npy",
                              "../DataBDF/OutNikodem/OutNikodem_val.npy",
                              "../DataBDF/OutNikodem/OutNikodem_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    batch = next(iter(val_data))
    sample = batch[0].cpu().detach().numpy()
    labels = batch[1].cpu().detach().numpy()
    for i in range(sample.shape[0]):
        label = labels_mapping[str(labels[i])]
        sample_1CH = sample[i, :, 0, :].squeeze()
        n_samples = sample_1CH.shape[0]
        total_duration = n_samples / sampling_frequency
        sampling_times = np.linspace(0, total_duration, n_samples)
        power, times, frequencies, coif = cwt_spectrogram(sample_1CH, sampling_frequency)
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(sampling_times, sample_1CH, color='b');

        ax1.set_xlim(0, total_duration)
        ax1.set_xlabel('time (s)')
        # ax1.axis('off')
        spectrogram_plot(power, times, frequencies, coif, cmap='jet', norm=LogNorm(), ax=ax2)

        ax2.set_xlim(0, total_duration)
        # ax2.set_ylim(0, 0.5*sampling_frequency)
        ax2.set_ylim(2.0 / total_duration, 0.5 * sampling_frequency)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('frequency (Hz)')
        fig.suptitle(label, fontsize=16)
        plt.savefig('../logs/spectrogram_images/spectrogram{}.png'.format(i))
        plt.close()