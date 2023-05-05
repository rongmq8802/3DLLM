'''
Author: Diantao Tu
Date: 2023-04-17 23:25:05
'''
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry

import logging 
import argparse
import numpy as np







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
    name="blip2_llama", model_type="blip2_3d_caption", is_eval=True, device=device
)


cloud_path = "/public/public_data/3DLLM/merge/train/scene0324_00.pth"
"房间里的餐桌和椅子。一张木桌上放着一把木椅。客厅里有一张桌子，窗户旁边有一扇窗户。房间里有两把椅子和一张桌子。一张带椅子和木制桌面的餐桌。房间里有两把椅子和一张桌"  "stage2_only"
"有一扇窗户和一把木椅的房间。一张桌子，桌子和椅子。一张有椅子和木桌的房间。一个有椅子和桌子的房间。一把木椅和一张木桌。一个房间有一张桌子和两把椅子。房间里的木桌和木椅"
cloud_path = "/public/public_data/3DLLM/merge/train/scene0073_03.pth"
"车库里摆满了各种东西的小房间。一只猫躺在床上，旁边有一盏灯。一只猫躺在床上，旁边有一盏灯。有桌子和梳妆台的小房间。桌子前的椅子上放着一台电脑。\n一把剪刀挂在一把吊伞"
"一只小狗站在白色的床上，床上有一个梳妆台。一个白色的冰箱，里面放着一些物品。有梳妆台和梳毛器的小房间。一张白色的床，上面有一个梳妆台。一张黑白的毛绒玩具熊图像的地板。\n带梳"
cloud_path = "/public/public_data/3DLLM/merge/train/Area_2/office_5.pth"
"书架上有很多书的房间。摆满了书和杂志的书架。一台笔记本电脑放在一堆书上。一张杂乱的桌子，上面放着一台电脑和一些书。有很多书和书架的图书馆。这是个办公室。\n杂乱的房"
"书架上有很多书的房间。摆满了书的书架。一张有书和书架的房间。有很多书和一台电脑的图书馆。有很多书和书架的书架。一张杂乱的桌子，上面放着一台电脑和一些书。这是个办公室"
cloud_path = "/public/public_data/3DLLM/merge/train/scene0191_00.pth"
cloud_path = "/public/public_data/3DLLM/merge/train/Area_1/hallway_1.pth"
cloud_path = "/public/public_data/3DLLM/merge/train/scene0604_00.pth"
cloud_path = "/public/public_data/3DLLM/4_23_rmq_sort_data/piano/scene0604_00_noceiling.pth"
cloud = torch.load(cloud_path)

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

cloud = vis_processors["eval"](cloud)


for k in cloud.keys():
    if(isinstance(cloud[k], torch.Tensor)):
        cloud[k] = cloud[k].to(device)
        cloud[k] = cloud[k].unsqueeze(0)

cloud_copy = cloud.copy()



cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维场景。"}, max_length=150, num_beams=1)
print(result)

cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维场景的结构布局。"}, max_length=100, num_beams=1)
print(result)

cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维点云。"}, max_length=100, num_beams=1)
print(result)


cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": " "}, max_length=100, num_beams=1)
print(result)

cloud = cloud_copy.copy()
result = model.generate_with_hidden_prompt({"cloud":cloud}, max_length=100, num_beams=1)
print(result)




print("============generate with prompt=============")
result = model.generate({"cloud":cloud, "text_input": "这是一个厕所"}, max_length=50, num_beams=1)
print(result)

cloud = cloud_copy.copy()
print("============generate without prompt=============")
result = model.generate({"cloud":cloud}, max_length=50, num_beams=1)
print(result)

cloud = cloud_copy.copy()
print("============generate with hidden prompt=============")
result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "这是一个厕所"}, max_length=50, num_beams=1)
print(result)

cloud = cloud_copy.copy()
print("============generate with hidden prompt=============")
result = model.generate_with_hidden_prompt({"cloud":cloud}, max_length=50, num_beams=1)
print(result)

import json
input_data_path = "/public/public_data/3DLLM/str3d_pth/ZN-train4.json"
pairs = []
with open(input_data_path, "r") as f:
    init_pairs = json.load(f)
    for key in init_pairs:
        if isinstance(init_pairs[key], str):
            pairs.append([key, init_pairs[key]])
        elif isinstance(init_pairs[key], list):
            pairs.append([key] + init_pairs[key])
        else:
            raise ValueError("Error: The value of key {} is not str or list".format(key))
output_data = {}
for i, item in enumerate(pairs):
    cloud_path = item[0]
    cloud = torch.load(cloud_path)
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

    cloud = vis_processors["eval"](cloud)
    for k in cloud.keys():
        if(isinstance(cloud[k], torch.Tensor)):
            cloud[k] = cloud[k].to(device)
            cloud[k] = cloud[k].unsqueeze(0)


    result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": item[2]}, max_length=100, num_beams=1, max_sentences = 2)
    output_data[cloud_path] = [item[1],result[0]]
    print("Finish {} / {}".format(i, len(pairs)))

with open("ZN-train4-predict.json", "w") as f:
    f.write(json.dumps(output_data, ensure_ascii=False, indent=1))