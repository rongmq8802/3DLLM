'''
Author: Diantao Tu
Date: 2023-04-15 19:16:02
'''
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import lavis.common.dist_utils as dist_utils
import logging
import os, json
import numpy as np
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

@registry.register_task("scanqa_pretrain")
class ScanQAPretrainTask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for key_dataset, value_dataset in datasets.items():
            for key_split, value_split in value_dataset.items():
                self.anno_files[key_split] = value_split.scanqa_text_pairs

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(self.anno_files), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            # ques_id = int(ques_id.item())
            ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, epoch):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_%d_vqa_result" % epoch,
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(val_result=val_result, epoch=epoch)

        return metrics

    def evals_json(self, gold_data, preds):
        score_list = ['Top1 (EM)']
        score = {s:[] for s in score_list}
        
        for ins in gold_data:
            question_id=ins['question_id']
            question=ins['question']
            ref_answers=ins['answers'] # 首先把答案取出来
        
            pred=preds[question_id] 

            # top-1
            answer = pred['answer'] # 取出预测结果
            if answer in ref_answers:
                score['Top1 (EM)'].append(1)
            else:
                score['Top1 (EM)'].append(0)
            
        rlt={}
        for k,v in score.items():
            assert len(v)==len(gold_data),len(v)
            rlt[k]=np.mean(v)*100
        return rlt


    def eval_pycoco(self, gts, preds_new, use_spice=False):

        score_list = ['Top1 (EM)','Top1 (F-value)','BLEU-1','BLEU-2','BLEU-3','BLEU-4']
        score = {s:[] for s in score_list}
        
        scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
        ]
        if use_spice:
            scorers.append((Spice(), "SPICE"))

        tokenizer = PTBTokenizer()
        # pycocoeval
        
        gts = {ins['question_id']:[{'caption':ans} for ans in ins['answers']] for ins in gts}
        res = {qid:[{'caption':value['answer']}] for qid,value in preds_new.items()}
        
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        
        # =================================================
        # Compute scores
        # =================================================
        rlt={}
        for scorer, method in scorers:
            #eprint('computing %s score...'%(scorer.method()))
        
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
            #       print("%s: %0.3f"%(m, sc*100))
                    rlt[m]=sc*100
            else:
            #  print("%s: %0.3f"%(method, score*100))
                rlt[method]=score*100
        return rlt


    @dist_utils.main_process
    def _report_metrics(self, val_result, epoch):
        gts = self.anno_files["val"]
        preds_new = {}
        for (q,g) in enumerate(gts):
            if q>=len(val_result):
                break
            preds_new[q] = {'answer':val_result[q]['answer']}
            gts[q]['question_id'] = q
        gts = gts[:len(preds_new)]

        score = self.evals_json(gts, preds_new)
        scores2 = self.eval_pycoco(gts, preds_new)
        score.update(scores2)
        metrics = score

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(f"Epoch {epoch}: {metrics}\n")

        return metrics


