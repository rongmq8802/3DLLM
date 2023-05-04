import torch 
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from typing import Dict
import numpy as np
import plyfile
import numpy as np
import json
from typing import List

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

device = torch.device("cuda")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_llama", model_type="blip2_3d_caption", is_eval=True, device=device
)


# 读取json文件
input_data_path = "/public/public_data/3DLLM/str3d_pth/ZN-train5.json"
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

# 对每一个点云都进行相应的描述，把结果保存在output_data中
with open("caption_v3.txt", "w") as f:
    output_data = {}
    idx = 0
    while(idx < len(pairs)):
        item = pairs[idx]
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


        result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": item[2]}, max_length=150, num_beams=1)
        result = post_process(result, max_sentences=2)
        output_data[cloud_path] = [item[1],result[0]]
        print("Finish {} / {}".format(idx, len(pairs)))
        f.write("{} \n {} \n {} \n \n".format(cloud_path, item[1], result[0]))
        f.flush()
        # structure 3d 有3179 个点云, 每隔 10个点云描述一次
        # 剩下的点云每间隔 5 个点云描述一次
        if(idx < 3179):
            idx += 4
        else:
            idx += 1

with open("caption_v3.json", "w") as f:
    f.write(json.dumps(output_data, ensure_ascii=False, indent=1))
