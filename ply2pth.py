import glob
import multiprocessing as mp
import pdb

import numpy as np
import plyfile
import torch
import os


def convert(input_file_path, output_file_path):
    a = plyfile.PlyData().read(input_file_path)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1
    w = np.zeros((coords.shape[0]), dtype=np.int64)
    torch.save((coords, colors, w), output_file_path)
    # torch.save({"coord": coords, "color": colors, "semantic_gt": w}, output_file_path)

input_cloud_base_folder = "/public/public_data/3DLLM/str3d_pc/pc/"
splits = ["test"]
output_cloud_base_folder = "/public/public_data/3DLLM/str3d_pth/pth/"
for split in splits:
    input_cloud_folder = os.path.join(input_cloud_base_folder, split)
    output_cloud_folder = os.path.join(output_cloud_base_folder, split)
    os.makedirs(output_cloud_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_cloud_folder, "*.ply"))

    # 使用多线程，同时显示目前已经处理的文件数量
    pool = mp.Pool(processes=30)
    print("start processing {}, cpu count: {}".format(split, 30))
    for i, input_file_path in enumerate(file_paths):
        output_file_path = os.path.join(output_cloud_folder, os.path.basename(input_file_path).replace(".ply", ".pth"))
        pool.apply_async(convert, (input_file_path, output_file_path))
        # convert(input_file_path, output_file_path)
        print("processing: ", i, " / ", len(file_paths))
    pool.close()
    pool.join()
    print("finish processing: ", split)

    

    
import torch
cloud_path = output_cloud_base_folder + "/train/scene_00000.pth"
cloud = torch.load(cloud_path)
print(cloud.keys())
print(cloud["coord"].shape)
print(cloud["color"].shape)
print(cloud["semantic_gt"].shape)