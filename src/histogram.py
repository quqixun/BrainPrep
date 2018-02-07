from __future__ import print_function

import os
import numpy as np
from tqdm import *
import nibabel as nib
from math import factorial
import matplotlib.pyplot as plt


def load_nii(path):
    return nib.load(path).get_data()


def compute_hist(volume, bins_num):
    bins = np.arange(0, bins_num)
    hist = np.histogram(volume, bins=bins, density=True)
    return hist[1][2:], hist[0][1:]


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def plot_hist(infos, bins_num):
    plt.figure()
    plt.title("Histogram of All Difference Volumes", fontsize=12)
    for info in tqdm(infos):
        path, label = info[0], info[1]
        color = "r" if label == "AD" else "b"
        volume = load_nii(path)
        x, y = compute_hist(volume, bins_num)
        non_zero_idx = np.where(y > 0)
        x = x[non_zero_idx]
        y = y[non_zero_idx]
        y = savitzky_golay(y, 9, 0)
        plt.plot(x, y, color, lw=0.3, alpha=0.5, label=label)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    plt.xlabel("Intensity", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid("on", linestyle="--", linewidth=0.5)
    plt.show()
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
data_src_dir = os.path.join(data_dir, "ADNIEnhance")
data_labels = ["AD", "NC"]

data_src_infos = []
for label in data_labels:
    src_label_dir = os.path.join(data_src_dir, label)
    for subject in os.listdir(src_label_dir):
        data_src_infos.append([os.path.join(src_label_dir, subject), label])

bins_num = 256
plot_hist(data_src_infos, bins_num)
