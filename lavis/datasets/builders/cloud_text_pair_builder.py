'''
Author: Diantao Tu
Date: 2023-04-15 20:24:37
'''
import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.cloud_text_pair_datasets import CloudTextPairDataset


@registry.register_builder("3d_caption")
class CloudCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = CloudTextPairDataset
    eval_dataset_cls = CloudTextPairDataset

    DATASET_CONFIG_DICT = {
        "default" : "configs/datasets/point_cloud/defaults_cap.yaml"
    }

    # 不需要下载数据，所以覆盖父类的方法
    def _download_data(self):
        return 
    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
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
                text_prompt="",
                path=ann_info[split].storage,
            )

        return datasets
