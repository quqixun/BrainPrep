from __future__ import print_function

import os
import numpy as np
import nibabel as nib
from scipy.signal import medfilt
from multiprocessing import Pool, cpu_count


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    nii = nib.load(path)
    return nii.get_data(), nii.get_affine()


def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return


def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)


def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume

    return volume


def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, normed=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def unwarp_enhance(arg, **kwarg):
    return enhance(*arg, **kwarg)


def enhance(src_path, dst_path, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    print("Preprocess on: ", src_path)
    try:
        volume, affine = load_nii(src_path)
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        save_nii(volume, dst_path, affine)
    except RuntimeError:
        print("\tFailed on: ", src_path)


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
data_src_dir = os.path.join(data_dir, "ADNIDenoise")
data_dst_dir = os.path.join(data_dir, "ADNIEnhance")
data_labels = ["AD", "NC"]
create_dir(data_dst_dir)

data_src_paths, data_dst_paths = [], []
for label in data_labels:
    src_label_dir = os.path.join(data_src_dir, label)
    dst_label_dir = os.path.join(data_dst_dir, label)
    create_dir(dst_label_dir)
    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        data_dst_paths.append(os.path.join(dst_label_dir, subject))

kernel_size = 3
percentils = [0.5, 99.5]
bins_num = 256
eh = True

# Test
# enhance(data_src_paths[0], data_dst_paths[0],
#         kernel_size, percentils, bins_num, eh)

# Multi-processing
subj_num = len(data_src_paths)
paras = zip(data_src_paths, data_dst_paths,
            [kernel_size] * subj_num,
            [percentils] * subj_num,
            [bins_num] * subj_num,
            [eh] * subj_num)
pool = Pool(processes=cpu_count())
pool.map(unwarp_enhance, paras)
