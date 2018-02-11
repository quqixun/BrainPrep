from __future__ import print_function

import os
import numpy as np
import nibabel as nib
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def load_nii(path):
    nii = nib.load(path)
    return nii.get_data(), nii.get_affine()


def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return


def extract_features(data):
    x_idx, y_idx, z_idx = np.where(data > 0)
    features = []
    for x, y, z in zip(x_idx, y_idx, z_idx):
        features.append([data[x, y, z], x, y, z])
    return np.array(features)


def kmeans_cluster(data, n_clusters):
    features = extract_features(data)
    intensities = features[..., 0].reshape((-1, 1))
    kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++",
                          precompute_distances=True, verbose=0,
                          random_state=7, n_jobs=1,
                          max_iter=1000, tol=1e-6).fit(intensities)

    labels = np.zeros(data.shape)
    for l, f in zip(kmeans_model.labels_, features):
        labels[int(f[1]), int(f[2]), int(f[3])] = l + 1

    return labels


def fuzzy_cmeans_cluster(data, n_clusters):
    features = extract_features(data)
    intensities = features[..., 0].reshape((1, -1))

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        intensities, n_clusters, 2, error=1e-6,
        maxiter=1000, init=None, seed=7)
    labels_ = np.argmax(u, axis=0)

    labels = np.zeros(data.shape)
    for l, f in zip(labels_, features):
        labels[int(f[1]), int(f[2]), int(f[3])] = l + 1

    return labels


def get_target_label(labels, data):
    labels_set = np.unique(labels)
    mean_intensities = []
    for label in labels_set[1:]:
        label_data = data[np.where(labels == label)]
        mean_intensities.append(np.mean(label_data))
    target_intensity = np.median(mean_intensities)  # GM
    # target_intensity = np.max(mean_intensities)  # WM
    # target_intensity = np.min(mean_intensities)  # CSF
    target_label = mean_intensities.index(target_intensity) + 1
    return target_label


def unwarp_segment(arg, **kwarg):
    return segment(*arg, **kwarg)


def segment(src_path, dst_path, labels_path=None, method="km"):
    print("Segment on: ", src_path)
    try:
        data, affine = load_nii(src_path)
        n_clusters = 3

        if method == "km":
            # Method 1 - KMeans
            labels = kmeans_cluster(data, n_clusters)
        elif method == "fcm":
            # Method 2 - Fuzzy CMeans
            labels = fuzzy_cmeans_cluster(data, n_clusters)

        target = get_target_label(labels, data)
        gm_mask = np.copy(labels).astype(np.float32)
        gm_mask[np.where(gm_mask != target)] = 0.333
        gm_mask[np.where(gm_mask == target)] = 1.
        data = data.astype(np.float32)
        gm = np.multiply(data, gm_mask)
        save_nii(labels, labels_path, affine)
        save_nii(gm, dst_path, affine)
    except RuntimeError:
        print("\tFalid on: ", src_path)

    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
data_src_dir = os.path.join(data_dir, "ADNIEnhance")
data_dst_dir = os.path.join(data_dir, "ADNIKMSegment")
data_labels = ["AD", "NC"]
create_dir(data_dst_dir)

data_src_paths, data_dst_paths, labels_paths = [], [], []
for label in data_labels:
    src_label_dir = os.path.join(data_src_dir, label)
    dst_label_dir = os.path.join(data_dst_dir, label)
    create_dir(dst_label_dir)
    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        subj_name = subject.split(".")[0]
        dst_subj_dir = os.path.join(dst_label_dir, subj_name)
        create_dir(dst_subj_dir)
        data_dst_paths.append(os.path.join(dst_subj_dir, subject))
        labels_paths.append(os.path.join(dst_subj_dir, subj_name + "_labels.nii.gz"))

method = "km"  # "fcm" or "km"

# Test
# segment(data_src_paths[0], data_dst_paths[0], labels_paths[0])

# Multi-processing
subj_num = len(data_src_paths)
paras = zip(data_src_paths, data_dst_paths,
            labels_paths, [method] * subj_num)
pool = Pool(processes=cpu_count())
pool.map(unwarp_segment, paras)
