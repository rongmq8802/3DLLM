'''
Author: RongMQ
Date: 2023-12-05 13:45:53
LastEditors: RongMQ
LastEditTime: 2023-12-05 19:33:15
Description: 
FilePath: /3DLLM_2/lavis/datasets/builders/scanqa_pair_builder.py
'''
import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.scanqa_datasets import ScanQAPairDatasets, ScanQAPairDatasets

@registry.register_builder("scanqa")
class ScanQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ScanQAPairDatasets
    eval_dataset_cls = ScanQAPairDatasets

    DATASET_CONFIG_DICT = {
        "default" : "configs/datasets/point_cloud/scanqa.yaml"
    }

    # 不需要下载数据，所以覆盖父类的方法
    def _download_data(self):
        return 
    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_info = build_info.pcdir
        datasets = dict()

        # split: train, val, test 
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            # processors 
            vis_processor = (self.vis_processors["train"] if is_train else self.vis_processors["eval"])
            text_processor = (self.text_processors["train"] if is_train else self.text_processors["eval"])

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor = vis_processor,
                text_processor = text_processor,
                ann_paths=ann_info[split].storage,
                vis_root=vis_info[split].storage
            )

        return datasets
