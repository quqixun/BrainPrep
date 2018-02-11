from __future__ import print_function

import os
import shutil
import subprocess
from multiprocessing import Pool, cpu_count


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def fast(src_path, dst_path, temp_dir, temp_path):
    command = ["fast", "-t", "1", "-n", "3", "-H", "0.1", "-I", "1", "-l", "20.0",
               "-o", temp_dir, src_path]
    subprocess.call(command, stdout=open(os.devnull), stderr=subprocess.STDOUT)
    shutil.copyfile(temp_path, dst_path)
    shutil.rmtree(os.path.dirname(temp_dir))
    return


def unwarp_segment(arg, **kwarg):
    return segment(*arg, **kwarg)


def segment(src_path, dst_path, temp_dir, temp_path):
    print("Segment on: ", src_path)
    try:
        fast(src_path, dst_path, temp_dir, temp_path)
    except RuntimeError:
        print("\tFalid on: ", src_path)
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
data_src_dir = os.path.join(data_dir, "ADNIDenoise")
data_dst_dir = os.path.join(data_dir, "ADNISegment")
data_labels = ["AD", "NC"]
create_dir(data_dst_dir)

data_src_paths, data_dst_paths = [], []
temp_dirs, temp_paths = [], []
for label in data_labels:
    src_label_dir = os.path.join(data_src_dir, label)
    dst_label_dir = os.path.join(data_dst_dir, label)
    create_dir(dst_label_dir)
    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        data_dst_paths.append(os.path.join(dst_label_dir, subject))
        subj_name = subject.split(".")[0]
        temp_dir = os.path.join(dst_label_dir, subj_name, subj_name)
        create_dir(os.path.dirname(temp_dir))
        temp_dirs.append(temp_dir)
        temp_paths.append(temp_dir + "_pve_1.nii.gz")

# Test
# print(data_src_paths[0], data_dst_paths[0],
#       temp_dirs[0], temp_paths[0])
# segment(data_src_paths[0], data_dst_paths[0],
#         temp_dirs[0], temp_paths[0])

# Multi-processing
subj_num = len(data_src_paths)
paras = zip(data_src_paths, data_dst_paths, temp_dirs, temp_paths)
pool = Pool(processes=cpu_count())
pool.map(unwarp_segment, paras)
