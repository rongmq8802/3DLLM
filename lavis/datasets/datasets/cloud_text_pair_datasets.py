'''
Author: Diantao Tu
Date: 2023-04-15 20:28:14
'''

import os
from collections import OrderedDict
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from typing import List, Tuple, Dict
import json
import numpy as np
import logging
import plyfile

def load_point_cloud(path:str) -> Dict[str, torch.Tensor]:
    """
    从文件中读取点云
    path: 点云路径,绝对路径
    return: 点云, 字典类型, 包含 "coord", "color", "semantic_gt" 三个key
    """
    file_type = path.split(".")[-1]
    if file_type == "pth":
        cloud = torch.load(path)
        # 专门针对strucutred3D数据集的处理, 因为保存的 pth 是个tuple的格式, 而且点云的尺度很大, 需要归一化
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
        max_value = np.max(coords)
        min_value = np.min(coords)
        final_value = max(abs(max_value), abs(min_value))
        # S3DIS/ScanNet/Structure3D三个数据集的尺度没有进行统一。
        # 尤其对于Structure3D数据集，尺度非常大，所以我们在这设置一个阈值，
        # 若final_value大于1000，则认为是Structure3D数据集，然后将点云的坐标归一化到[-5, 5]的范围内
        # 对于智能化3层的场景, 我已经预先进行了归一化, 所以这里不会触发这个if语句
        if final_value > 1000:
            cloud["coord"] = coords / final_value * 5.0
        else:
            cloud["coord"] = coords
        cloud["color"] = colors
        cloud["semantic_gt"] = semantic_gt
    else:
        raise ValueError("file type {} not supported".format(file_type))
    
    return cloud

def load_pairs(path:str) -> List[Tuple[str, str]]:
    """
    从文件中读取点云和对应的caption, 以及可能存在的prompt
    输入的文件格式为json,具体实例如下:
    "path/to/pointcloud1.pth": ["This is caption 1", "This is prompt 1"], 
    "path/to/pointcloud2.pth": ["This is caption 2", "This is prompt 2"]
    "path/to/pointcloud3.pth": ["This is caption 3"]
    "path/to/pointcloud4.pth": "This is caption 4"
    
    path: 文件路径, 这是一个绝对路径
    return: List[List[str, str, str]], 每个元素是一个List, 第一个元素是点云路径, 第二个元素是对应的caption, 第三个元素是prompt(如果存在的话)
    """
    pairs = []
    with open(path, "r") as f:
        init_pairs = json.load(f)
        for key in init_pairs:
            if isinstance(init_pairs[key], str):
                pairs.append([key, init_pairs[key]])
            elif isinstance(init_pairs[key], list):
                pairs.append([key] + init_pairs[key])
            else:
                raise ValueError("Error: The value of key {} is not str or list".format(key))
    return pairs

# 虽然这里继承了 BaseDataset, 但这个类覆写了__getitem__和__len__方法, 这里继承的主要目的是之后dataloader建立的时候不报错
# 由于时间关系, 我没有看dataloader是怎么建立的, 所以就直接继承了BaseDataset
class CloudTextPairDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, text_prompt:str="", path:str=""):
        """
        点云和对应的caption的数据集, 里面存储了点云和对应的caption, 如果点云对应于N个caption, 那么就会有N个元素具有相同的点云不同的caption
        vis_processor: 对点云进行处理的函数, 叫做 vis_processor 只是为了和原本的代码保持一致
        text_processor: 对caption进行处理的函数
        text_prompt: caption的前缀, 最终的caption是 text_prompt + caption
        path: 存储点云和对应的caption的文件的路径, 绝对路径
        """
        super().__init__()
        self.text_prompt = text_prompt
        self.vis_processor = vis_processor  # 其实是对点云进行处理, 只是名字叫做 vis_processor
        self.text_processor = text_processor
        self.cloud_text_pairs = load_pairs(path)
        # self.cloud_text_pairs = [["test.pcd", "frist test caption"],["test2.pcd", "second test caption"]]
        
        

    def __getitem__(self, index):

        pair = self.cloud_text_pairs[index]
        cloud = load_point_cloud(pair[0])
        caption = pair[1]

        cloud = self.vis_processor(cloud)
        caption = self.text_processor(caption)

        data =  {"cloud": cloud, "text_input": caption, "cloud_path": pair[0]}
        if(len(pair) > 2):
            data["prompt"] = pair[2]
        return data

        image = torch.ones((3, 224, 224))

        return {"image": image, "text_input": caption}

    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )
    
    def __len__(self):
        return len(self.cloud_text_pairs)


if __name__ == "__main__":
    path = "/data3/rmq/points_text_datasets/S3DIS/text_embed/ZN-s3dis_text_absolute.json"
    cloud_text_pairs = load_pairs(path)
    print("true")