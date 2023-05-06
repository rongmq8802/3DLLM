'''
Author: Diantao Tu
Date: 2023-05-06 10:57:50
'''
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
import plyfile
import logging 
import argparse
import numpy as np
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




logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

# 从命令行读取参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="")
args = parser.parse_args()

device = torch.device("cuda")
device = torch.device(args.device)


model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_chatglm", model_type="blip2_3d_caption_chatglm", is_eval=True, device=device
)


cloud_path = "/data3/lyz/merge/train/scene0191_00.pth"

cloud = torch.load(cloud_path)
cloud = vis_processors["eval"](cloud)

for k in cloud.keys():
    if(isinstance(cloud[k], torch.Tensor)):
        cloud[k] = cloud[k].to(device)
        cloud[k] = cloud[k].unsqueeze(0)

cloud_copy = cloud.copy()

cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维场景。"}, max_length=150, num_beams=1)
print(result)