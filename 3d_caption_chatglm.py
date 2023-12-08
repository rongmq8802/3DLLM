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
from point_cloud_loader import load_point_cloud



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


model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_chatglm", model_type="blip2_3d_caption_chatglm", is_eval=True, device=device
)


cloud_path = "/public/public_data/3DLLM/4_23_rmq_sort_data/piano/scene0604_00_noceiling.pth"

# cloud = torch.load(cloud_path)
cloud = load_point_cloud(cloud_path)
cloud = vis_processors["eval"](cloud)


for k in cloud.keys():
    if(isinstance(cloud[k], torch.Tensor)):
        cloud[k] = cloud[k].to(device)
        cloud[k] = cloud[k].unsqueeze(0)

cloud_copy = cloud.copy()

cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维场景。"}, max_length=100, num_beams=1)
print(result)