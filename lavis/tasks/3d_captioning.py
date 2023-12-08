'''
Author: Diantao Tu
Date: 2023-04-17 18:27:30
'''
import json
import os
import torch
from lavis.common.dist_utils import main_process, is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from typing import Dict, List, Tuple

@registry.register_task("3d_captioning")
class CloudCaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
        )
    
    # 这个的验证就是把点云和描述拼接起来, 然后用beam search生成描述
    def evaluation(self, model, data_loader, cuda_enabled=True) -> List[Dict]:
        results = []
       
        for samples in data_loader:
            
            captions = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )

            cloud_paths = samples["cloud_path"]
            input_texts = samples["input_text"]
            for caption, cloud_path, input_text in zip(captions, cloud_paths, input_texts):
                results.append({"cloud_path": cloud_path, "input_text": input_text, "caption": caption})
        return results
    
    def after_evaluation(self, val_result: List[Dict], split_name:str, epoch:int, **kwargs) -> Dict[str, float]:
        # 保存结果
        if is_main_process():
            result_dir = registry.get_path("result_dir")
            result_path = os.path.join(result_dir, f"3d_caption_epoch_{epoch}_{split_name}.txt")
            with open(result_path, "w") as f:
                for result in val_result:
                    f.write(json.dumps(result) + "\n")
                
        return {"agg_metrics" : 0}