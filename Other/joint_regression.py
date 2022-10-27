#### Module used to calculate channel-wise joint regression coefficients. 
#### Joint regression coefficients can be treated as features for classification model.
### Original paper https://doi.org/10.1007/s00521-012-1244-3
from sklearn import preprocessing
from re import S
from matplotlib.lines import segment_hits
import numpy as np
from scipy.signal import periodogram
from numpy.linalg import inv
from torch import float32
from tqdm import tqdm
class JointRegression():
    
    def __init__(self, past_samples_num, channels_num, freqs):
        self.m = past_samples_num
        self.channels_num = channels_num
        self.freqs = freqs
    def __call__(self, data):
        #data dims (segments, channels, samples)
        if len(self.freqs) == 0:
            Vt = self.calculateVt(data)
            Vt = Vt.reshape((Vt.shape[0], -1))
            features = Vt
        elif self.m != 0:
            Vf = self.calculateVf(data)
            Vf = Vf.reshape((Vf.shape[0], -1))
            features = Vf
        else:
            Vt = self.calculateVt(data)
            Vt = Vt.reshape((Vt.shape[0], -1))
            Vf = self.calculateVf(data)
            Vf = Vf.reshape((Vf.shape[0], -1))
            features = np.concatenate((Vt, Vf), axis=1)
        features_normalized = self.standarize_features(features)
        return features_normalized
    def standarize_features(self, features):
        mean = np.mean(features, axis = 0)
        std = np.std(features, axis = 0) 
        normalized_features = (features - mean)/std
        return normalized_features
    def calculateVt(self, data):
        segments_num = data.shape[0]
        channels_num = data.shape[1]
        samples = data.shape[2]
        L = np.zeros((samples - self.m, self.m*channels_num))
        Y = np.zeros((samples - self.m))
        C = np.zeros((self.m*channels_num))
        Vt =  np.zeros((segments_num, channels_num, self.m*channels_num))
        segments = tuple(np.split(data, segments_num, axis=0))
        for i, segment in enumerate(tqdm(segments)):
            segment = segment.reshape((channels_num, -1))
            #channels = tuple(np.split(segment, segment.shape[0], axis=0))
            for k in range(samples - self.m):
                L[k, :] = segment[:, k:(self.m + k)].flatten()
            for k in range(channels_num):
                Y = segment[k, self.m:]
                C = np.matmul(np.transpose(L), L)
                C = inv(C)
                C = np.matmul(C, np.transpose(L))
                C = np.matmul(C, Y)
                Vt[i, k, :] = C
        return Vt
    def calculateVf(self, data):
        segments_num = data.shape[0]
        channels_num = data.shape[1]
        samples = data.shape[2]
        P = np.zeros((segments_num, channels_num, len(self.freqs)))
        segments = tuple(np.split(data, segments_num, axis=0))
        for i, segment in enumerate(tqdm(segments)):
            segment = segment.reshape((channels_num, -1))
            channels = tuple(np.split(segment, channels_num, axis=0))
            for k, channel in enumerate(channels):
                channel = channel.reshape((samples))
                for l, freq in enumerate(self.freqs):
                    omega = 2*np.pi*freq
                    W = np.exp(1j*(-1)*2*np.pi*omega/samples)
                    X = sum(channel*W)
                    magnitude = np.sqrt(np.real(X)**2 + np.imag(X)**2)
                    P[i, k, l] = (1/samples)*magnitude
        return P
