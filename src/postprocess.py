from __future__ import print_function

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


def plot_middle(data, slice_no=None):
    if not slice_no:
        slice_no = data.shape[-1] // 2
    plt.figure()
    plt.imshow(data[..., slice_no], cmap="gray")
    plt.show()
    return


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def load_nii(path):
    return nib.load(path).get_data()


def save_nii(data, path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return


def brain_mask(data, mask):
    return np.multiply(data, mask)


def resize(data, target_shape=[96, 112, 96]):
    factor = [float(t) / float(s) for t, s in zip(target_shape, data.shape)]
    resized = zoom(data, zoom=factor, order=1, prefilter=False)
    return resized


def norm(data):
    # obj_idx = np.where(data > 0)
    # obj = data[obj_idx]
    # obj_mean, obj_std = np.mean(obj), np.std(obj)
    # obj = (obj - obj_mean) / obj_std
    # data[obj_idx] = obj
    data = data / float(np.max(data))
    return data


def unwarp_postprocess(arg, **kwarg):
    return postprocess(*arg, **kwarg)


def postprocess(src_path, dst_path, temp_path=None, is_mask=False):
    print("Wroking on: ", src_path)
    try:
        data = load_nii(src_path)
        if is_mask:
            mask = load_nii(temp_path)
            data = brain_mask(data, mask)
        data = resize(data)
        # data = norm(data)
        save_nii(data, dst_path)
    except RuntimeError:
        print("\tFalid on: ", src_path)


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
data_src_dir = os.path.join(data_dir, "ADNISegment")
data_dst_dir = os.path.join(data_dir, "ADNISegmentPost")
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

temp_path = os.path.join(data_dir, "Template", "bianca_exclusion_mask.nii.gz")

# Test
# postprocess(data_src_paths[0], data_dst_paths[0], temp_path)

# Multi-processing
is_mask = False
subj_num = len(data_src_paths)
paras = zip(data_src_paths, data_dst_paths,
            [temp_path] * subj_num, [is_mask] * subj_num)
pool = Pool(processes=cpu_count())
pool.map(unwarp_postprocess, paras)
