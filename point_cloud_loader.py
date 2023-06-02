import numpy as np
import torch
import plyfile
from typing import List, Tuple, Dict

def load_point_cloud(path:str) -> Dict[str, torch.Tensor]:
    """
    从文件中读取点云
    path: 点云路径,绝对路径
    return: 点云, 字典类型, 包含 "coord", "color", "semantic_gt" 三个key
    """
    file_type = path.split(".")[-1]
    if file_type == "pth":
        cloud = torch.load(path)
        if(isinstance(cloud, tuple)):
            cloud = {"coord": cloud[0], "color": cloud[1], "semantic_gt": cloud[2]}
            cloud["color"] = ((cloud["color"] + 1) * 127.5).astype(np.uint8)
            cloud["color"] = cloud["color"].astype(np.float64)
            cloud["coord"] = cloud["coord"].astype(np.float64)
            # 把 coord 中的值归一化到 [-5, 5] 之间
            max_value = np.max(cloud["coord"])
            min_value = np.min(cloud["coord"])
            final_value = max(abs(max_value), abs(min_value))
            cloud["coord"] = cloud["coord"] / final_value  * 5.0

        # "coord" "color" "semantic_gt"
        if "semantic_gt" in cloud.keys():
            cloud["semantic_gt"] = cloud["semantic_gt"].reshape([-1])
            cloud["semantic_gt"] = cloud["semantic_gt"].astype(np.int64)
    elif file_type == "ply":
        cloud = {}
        plydata = plyfile.PlyData().read(path)
        points = np.array([list(x) for x in plydata.elements[0]])
        coords = np.ascontiguousarray(points[:, :3]).astype(np.float64)
        colors = np.ascontiguousarray(points[:, 3:6]).astype(np.float64)
        semantic_gt = np.zeros((coords.shape[0]), dtype=np.int64)
        cloud["coord"] = coords
        cloud["color"] = colors
        cloud["semantic_gt"] = semantic_gt
    else:
        raise ValueError("file type {} not supported".format(file_type))
    
    return cloud