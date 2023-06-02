'''
Author: Diantao Tu
Date: 2023-04-24 18:24:23
'''
import torch 
import shutil
import tempfile
import requests
import os
import time

from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from flask import request
from flask_api import FlaskAPI
from typing import Dict, List
import numpy as np
import plyfile

app = FlaskAPI(__name__)

device = torch.device("cuda")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_llama", model_type="blip2_3d_caption", is_eval=True, device=device
)
# del _

# model = model.to(device)


def save_url_to_file(url):
    response = requests.get(url)
    tmpdir = tempfile.mkdtemp()
    file_name = os.path.basename(url)
    file_path = os.path.join(tmpdir, file_name)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path


def load_point_cloud(path_url: str) -> Dict[str, torch.Tensor]:
    """
    从文件中读取点云
    path: 点云路径,绝对路径
    return: 点云, 字典类型, 包含 "coord", "color", "semantic_gt" 三个key
    """
    if path_url and path_url.startswith('http'):
        path = save_url_to_file(path_url)
    else:
        path = path_url
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
        if final_value > 1000:
            cloud["coord"] = coords / final_value * 5.0
        else:
            cloud["coord"] = coords
        cloud["color"] = colors
        cloud["semantic_gt"] = semantic_gt
    else:
        raise ValueError("file type {} not supported".format(file_type))
    if path_url and path_url.startswith('http'):
        shutil.rmtree(os.path.dirname(path))
    return cloud


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data
    
def post_process(text_list:List[str], max_sentences:int = 2) -> List[str]:
    processed_text = []
    for text in text_list:
        text = text.replace("\n", "")       # 去掉换行符
        text = text.replace(" ", "")        # 去掉空格
        text_split = text.split("。")       # 按照句号分割
        # 如果只分割出一句话，说明这一段话中一个句号都没有，那么就按照逗号分割，
        # 并且只保留最后一个逗号之前的内容，如果连逗号都没有，那么我就无能为力了，这种情况下函数会直接返回空列表
        if len(text_split) == 1: 
            text_split = text.split("，")
            text_split = text_split[:-1]
            text_split = "，".join(text_split)
        else:
            text_split = text_split[:-1]
            text_split = text_split[:max_sentences] if len(text_split) > max_sentences else text_split
            text_split = "。".join(text_split)
        processed_text.append(text_split + "。")
    return processed_text

'''
传入的POST需要包含以下几项内容:
cloud_path : 点云的绝对路径
prompt : 对应的提问 
max_sentences : 生成的语句的最大句子数(可选)，这是因为生成的句子太多了效果就不好了。 default = 2
max_length : 生成的语句的最大长度(可选)。 default = 100
'''
@app.route("/caption", methods=['GET','POST'])
def caption():
    global device, model, vis_processors
    if(request.method == 'POST'):
        data = request.data
        if "cloud_path" not in data:
            return {"error": "cloud_path not found"}, 400
        cloud_path = data["cloud_path"]
        prompt = data["prompt"] if "prompt" in data else "请描述一下这个三维场景。"
        max_sentences = data["max_sentences"] if "max_sentences" in data else 2
        max_length = data["max_length"] if "max_length" in data else 100
        
        start_time = time.time()
        cloud = load_point_cloud(cloud_path)
        print(f"文件加载时间：{time.time() - start_time}")
        start_time = time.time()
        cloud = vis_processors["eval"](cloud)

        for k in cloud.keys():
            if(isinstance(cloud[k], torch.Tensor)):
                cloud[k] = cloud[k].to(device)
                cloud[k] = cloud[k].unsqueeze(0)
        print(f"文件预处理时间：{time.time() - start_time}")
        start_time = time.time()

        result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": prompt}, max_length=max_length, num_beams=1)
        result_processed = post_process(result, max_sentences=max_sentences)
        print(f"推理时间：{time.time() - start_time}")
        return_data = {"text" : result_processed}
        # 把结果和输入的数据一起返回
        return {"status": 1, "data": return_data, "message" : "success"}
 
    elif(request.method == 'GET'):
        return {"error": "nothing to GET"}

@app.route("/test")
def test():
    return {"test": "test_test"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
