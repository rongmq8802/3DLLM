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

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

def load_scanqa(path:str):
    with open(path, "r") as f:
        dataset = json.load(f)
    return dataset

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

class ScanQAPairDatasets(BaseDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=""):
        """
        """
        super().__init__()
        self.vis_root = vis_root
        self.vis_processor = vis_processor  # 其实是对点云进行处理, 只是名字叫做 vis_processor
        self.text_processor = text_processor
        self.scanqa_text_pairs = load_scanqa(ann_paths)
        
    # 这个地方需要引入一个文件夹的变量
    def __getitem__(self, index):

        pair = self.scanqa_text_pairs[index]
        cloud = load_point_cloud(os.path.join(self.vis_root, pair["scene_id"]+".pth"))
        question = pair["question"]
        answers = pair["answers"]
        answer_weight = {} # 用于存储每个答案的权重
        question_id = pair["question_id"]
        for answer in answers:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(answers)
            else:
                answer_weight[answer] = 1 / len(answers)
        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        cloud = self.vis_processor(cloud)
        question = self.text_processor(question)

        data = {"cloud": cloud, 
                "text_input": question, 
                "answer": answers,
                "weight": weights,
                "question_id": index}
        
        return data

    def collater(self, samples):

        cloud_list, question_list, answer_list, weight_list, question_id_list = [], [], [], [], []
        num_answers = []

        for sample in samples:
            cloud_list.append(sample["cloud"])
            question_list.append(sample["text_input"])
            weight_list.extend(sample["weight"])
            answers = sample["answer"]
            answer_list.extend(answers)
            num_answers.append(len(answers))
            question_id_list.append(sample["question_id"])
        
        cloud = {}
        for i in range(len(cloud_list)):
            for key, value in cloud_list[i].items():
                if key not in cloud.keys():
                    cloud[key] = []
                cloud[key].append(value)
        for key, value in cloud.items():
            cloud[key] = torch.stack(value, dim=0)

        data = {
            "cloud": cloud,
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
            "question_id": question_id_list
        }
        return data

    def __len__(self):
        # print(len(self.scanqa_text_pairs))
        return len(self.scanqa_text_pairs)
        # return 4000