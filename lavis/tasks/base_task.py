"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
import random


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

        '''
        cloud_path  ---- text_input 
                    ---- output_each_epoch  ---- 0 ---- text 
                                            ---- 1 ---- text
                                            ---- 2 ---- text
        以上是 result_each_sample 的结构，其中位于前面的是 key, 后面的是value 而每一个value 又是一个字典 
        
        '''

        self.result_each_sample = None

    # 给result_each_sample 初始化值
    def prepare_result(self, sample):
        if self.result_each_sample is None:
            self.result_each_sample = {}
        cloud_path_list = sample["cloud_path"]      # cloud_path_list 是一个列表，因为里面是一个batch 的数据
        text_input_list = sample["text_input"]      # text_input_list 是一个列表，因为里面是一个batch 的数据
        for (path, text_input) in zip(cloud_path_list, text_input_list):
            if path in self.result_each_sample:
                continue
            self.result_each_sample[path] = {"text_input" : text_input,
                                             "output_each_epoch": {}
                                             }  # output_each_epoch 是一个字典

    # 把前向传播的结果保存到 result_each_sample 中
    # results 是当前batch中每个样本解码得到的语言结果
    def add_result(self, sample, results, epoch):
        cloud_path_list = sample["cloud_path"]
        for (path, result) in zip(cloud_path_list, results):
            self.result_each_sample[path]["output_each_epoch"][epoch] = result

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        # # 如果是第二阶段的训练，那么ret_val 中应该包含 loss 以及 output_text
        # ret_val = model(samples)
        # return ret_val
        loss = model(samples)
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        result_dir = registry.get_path("result_dir")
        output_log_file_path = os.path.join(str(result_dir), "output_epoch_{}.txt".format(epoch))

        with open(output_log_file_path, "w") as f:

            for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
                # if using iter-based runner, we stop after iters_per_epoch iterations.
                if i >= iters_per_epoch:
                    break

                samples = next(data_loader)

                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                samples.update(
                    {
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    }
                )

                lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    ret_val = self.train_step(model=model, samples=samples)
                    loss = ret_val["loss"]

                # after_train_step()
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # update gradients every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()                     
                    else:    
                        optimizer.step()
                    optimizer.zero_grad()

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                # logging.info("loss: {}".format(loss.item()))
                if "output_text" in ret_val:
                    # logging.info("output_text: {}".format(ret_val["output_text"]))
                    # if(is_main_process()):
                    #     output_text = ret_val["output_text"]
                    #     input_text = samples["text_input"][0]
                    #     cloud_path = samples["cloud_path"][0]
                    #     line = "epoch: {}, iter: {} \n cloud: {} \n input: {} \n output: {} \n".format(
                    #         epoch, i, cloud_path, input_text, output_text
                    #     )
                    #     f.write(line + "\n")
                    self.prepare_result(sample=samples)
                    self.add_result(sample=samples, results=ret_val["output_text"], epoch=epoch)

            

            # 把 result_each_sample 保存到文件中
            if self.result_each_sample is not None:
                for cloud_path in self.result_each_sample:
                    # 生成一个0-1的随机数, 如果大于0.9, 则保存
                    if random.random() < 0.9:
                        continue
                    f.write("cloud: " + cloud_path + "\n")
                    f.write("text_input: " + self.result_each_sample[cloud_path]["text_input"] + "\n")
                    for epoch, result in self.result_each_sample[cloud_path]["output_each_epoch"].items():
                        f.write("epoch: {}, text_output: {} \n".format(epoch, result))
                    f.write("\n")

            # after train_epoch()
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))
            return {
                k: "{:.3f}".format(meter.global_avg)
                for k, meter in metric_logger.meters.items()
            }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
