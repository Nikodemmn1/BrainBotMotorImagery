import socket
import struct
import random
import pickle
import Server.server_data_convert as dc
import numpy as np
import torch
from Server.server_params import *


def load_mean_std():
    """Loading the file containing the pickled mean and std of the training dataset."""
    with open("./mean_std.pkl", "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_model():
    """Loading the trained OneDNet model"""
    model = torch.load("./model.pt")
    model.eval()
    return model


def load_file():
    with open('./testdata.pkl', 'rb') as handle:
        return pickle.load(handle)


def main():
    model = load_model()
    mean_std = load_mean_std()
    data = load_file()
    overlap = 128
    buffer_len = 3200
    data2 = [[] for _ in range(3)]
    buffer_duplicate = np.zeros((16, buffer_len)).astype('float32')
    for label, labeled_snippets in enumerate(data):
        for snippet in labeled_snippets:
            buffer = np.zeros((16, buffer_len)).astype('float32')
            buffer_filled = 0
            for mini_snippet_i in range(len(snippet[0])):
                mini_snippet = snippet[0][mini_snippet_i]
                mini_snippet_mean = snippet[1][mini_snippet_i][:, None]
                buffer = np.roll(buffer, -overlap, 1)
                buffer[:, -overlap:] = mini_snippet
                if buffer_filled + overlap < buffer_len:
                    buffer_filled += overlap
                else:
                    buffer_duplicate[:, ...] = buffer
                    buffer -= mini_snippet_mean
                    buffer *= 0.03125
                    data2[label].append(buffer)
                    buffer = buffer_duplicate
    min_class_list_len = min([len(class_list) for class_list in data2])

    ok = 0
    not_ok = 0

    for i in range(min_class_list_len):
        for j in range(3):
            buffer = data2[j][i]
            x = dc.prepare_data_for_classification(buffer, mean_std["mean"], mean_std["std"])
            x = x[:, :, 5:, :]
            y = dc.get_classification(x, model)
            out_ind = np.argmax(y.numpy())
            if out_ind == j:
                ok += 1
            else:
                not_ok += 1

    print(ok)
    print(not_ok)
    print(ok/(ok+not_ok))


if __name__ == '__main__':
    main()
