'''
Author: Diantao Tu
Date: 2023-04-16 13:46:07
'''
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModel
import time
from typing import List

@registry.register_model("blip2_chatglm")
class Blip2ChatGLM(Blip2Base):
    """
    BLIP2 ChatGLM model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "blip2_3d_stage2_chatglm": "configs/models/blip2_3d/blip2_3d_stage2_chatglm.yaml",          # 二阶段训练
        "blip2_3d_caption_chatglm": "configs/models/blip2_3d/blip2_3d_caption_chatglm.yaml",        # 点云描述
    }

    def __init__(
        self,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        point_cloud_encoder_model = None,
        freeze_point_cloud_encoder = True,
        max_cloud_size = 10000,
        num_query_token=32,
        chatglm_model_path="",
        prompt=None,
        max_txt_len=32,
        qformer_encoder_layer=12,
        point_cloud_encoder_pretrain_model_path = None,
    ):
        super().__init__()

        # 有时候会开好几个terminal，每个里面都运行了一个Blip2llama，为了区分它们，就用一个路径来区分
        self.model_path = None

        self.bert_tokenizer = None
        self.bert_model = None
        self.sim_threshold = 0.8
     
        self.cloud_encoder, self.ln_cloud = self.init_cloud_encoder(
            point_cloud_encoder_model, max_cloud_size, drop_path_rate, use_grad_checkpoint, point_cloud_encoder_pretrain_model_path
        )
        if freeze_point_cloud_encoder:
            for name, param in self.cloud_encoder.named_parameters():
                param.requires_grad = False
            self.cloud_encoder = self.cloud_encoder.eval()
            self.cloud_encoder.train = disabled_train
            logging.info("freeze point cloud encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.cloud_encoder.enc_channels[-1], encoder_layer=qformer_encoder_layer
        )

        # 把一部分网络固定删去了, 不理解为什么
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        t1 = time.time()
        # transformers 的读取代码
        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(chatglm_model_path, use_fast=False)
        self.chatglm_model = AutoModel.from_pretrained(chatglm_model_path, torch_dtype=torch.float16)
        # 这里设置为True, 是为了避免 model.generate() 函数报错, 不知道为什么要这么做
        self.chatglm_model.config.use_cache = True
        # self.llama_model = None
        logging.info("load llama model spend time: {:.4f} s".format(time.time() - t1))
         
        if self.chatglm_model is not None:
            for name, param in self.chatglm_model.named_parameters():
                param.requires_grad = False
        # 给的 llama-Chinese 模型中使用的是 </s> 作为结束符
        self.eos_token_id = self.chatglm_tokenizer.eos_token_id

        # 增加 pad token 不然二阶段会报错 ValueError: Asking to pad but the tokenizer does not have a padding token.
        # self.chatglm_tokenizer.pad_token = self.chatglm_tokenizer.eos_token 

        self.chatglm_proj = nn.Linear(self.Qformer.config.hidden_size, 
                                  self.chatglm_model.config.hidden_size if self.chatglm_model is not None else 4096 
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt


    def set_model_path(self, path:str):
        self.model_path = path

    def reload_from_checkpoint(self, num_qformer_layer:int,checkpoint_path:str):
        del self.Qformer, self.query_tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.cloud_encoder.enc_channels[-1], encoder_layer=num_qformer_layer
        )
        self.load_checkpoint(checkpoint_path)
        self.set_model_path(checkpoint_path)

    def comupte_simililiarity(self, text:List[str], device)->float:
        if(len(text) <= 1):
            return 0.0
        # 用于后处理生成的文字的参数
        if(self.bert_model is None):
            logging.info("load bert model")
            # self.bert_tokenizer = AutoTokenizer.from_pretrained("/public/public_data/3DLLM/pretrained_model/bert-base-chinese/")
            # self.bert_model = AutoModel.from_pretrained("/public/public_data/3DLLM/pretrained_model/bert-base-chinese/")
            # 不知道为什么, 从文件读取的bert模型有问题, 但文件里的bert模型就是先用以下方式得到, 然后保存的
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.bert_model = AutoModel.from_pretrained("bert-base-chinese")

            self.bert_model.to(device)
            self.bert_model.eval()
        
        # 找到最短的句子的长度
        min_len = min([len(t) for t in text])
        inputs = self.bert_tokenizer(text, max_length=min_len, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            text_vec1 = outputs.last_hidden_state[0, 0, :]
            text_vec2 = outputs.last_hidden_state[1, 0, :]
            simililarity = torch.cosine_similarity(text_vec1, text_vec2, dim=0)
        return simililarity.item()

    def postprocess_text(self, text:List[str], device):
        output = []
        # 把所有句子按照句号分割
        text_split = [t.split("。") for t in text]
        for split_each_text in text_split:
            # 如果只分割出来一个句子，那么就直接加入到输出中
            if(len(split_each_text) <= 1):
                output.append(split_each_text[0])
                continue
            # 如果有多个句子，那么就匹配连续两个句子的相似度 
            reference_id = 0
            available_split = [split_each_text[0]]      # 用来存储可用的句子，第一句肯定是可用的
            for curr_id in range(1, len(split_each_text)):
                # 如果当前句子长度小于等于1，那么就跳过, 因为这种句子不是完整的句子或者就是一个标点符号甚至是空字符串
                if(len(split_each_text[curr_id]) <= 1):
                    continue
                sim = self.comupte_simililiarity([split_each_text[reference_id], split_each_text[curr_id]], device)
                if sim <= self.sim_threshold:
                    available_split.append(split_each_text[curr_id])
                    reference_id = curr_id
            output.append("。".join(available_split))
        return output


    def forward(self, samples):
        image = samples["cloud"]
        device = image["coord"].device
        with self.maybe_autocast():
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)
            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # 这一行就相当于是把 learnable query 和 image 特征进行了交叉注意力
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # query_output.last_hidden_state [B , 32, 768] -> [B, 32, 5120]
        inputs_chatglm = self.chatglm_proj(query_output.last_hidden_state)
        atts_chatglm = torch.ones(inputs_chatglm.size()[:-1], dtype=torch.long).to(device)  # [B, 32]

        self.chatglm_tokenizer.padding_side = "right"
        
        # 如果输入的数据中自带prompt，那么就用自带的prompt，每条数据自带的prompt可以不同
        # 如果输入的数据中没有prompt，那么就用模型的默认prompt，每条数据都用同一个prompt
        if "prompt" in samples:
            prompt = samples["prompt"]
        else:
            prompt = [self.prompt] * image_embeds.shape[0]
        
        prompt_token = self.chatglm_tokenizer(prompt, return_tensors="pt", padding="longest").to(device)
        # tokenizer 会自动在输入的文本前面加上 bos token, 所以这里要减去1
        prompt_attention_mask = prompt_token.attention_mask[:, 1:]
        prompt_token.input_ids = prompt_token.input_ids[:, 1:]
        prompt_targets =  (torch.ones(prompt_attention_mask.size(), dtype=torch.long).to(device).fill_(-100))
        prompt_embeds = self.chatglm_model.model.embed_tokens(prompt_token.input_ids)

        # llama_tokenizer 会自动在输入的文本前面加上 bos token, 所以这里要减去1
        end_text = [t + "\n" for t in samples["text_input"]]
        end_token = self.chatglm_tokenizer(end_text, return_tensors="pt", padding="longest",
                                            truncation=True, max_length=self.max_txt_len).to(device)
        end_token.input_ids = end_token.input_ids[:, 1:]
        end_attention_mask = end_token.attention_mask[:, 1:]
        # 把padding 的位置设置为 -100, 这样就不会计算loss
        end_targets = end_token.input_ids.masked_fill(
            end_token.input_ids == self.chatglm_tokenizer.pad_token_id, -100
        )
        end_embeds = self.chatglm_model.model.embed_tokens(end_token.input_ids)


        # [B, 32]
        empty_targets = (
            torch.ones(atts_chatglm.size(), dtype=torch.long).to(device).fill_(-100)
        )
        # query + prompt + input_text
        targets = torch.cat([empty_targets, prompt_targets, end_targets], dim=1)    # [B, 32] cat [B, N]  -> [B, 32+N]
        inputs_embeds = torch.cat([inputs_chatglm, prompt_embeds, end_embeds], dim=1)
        attention_mask = torch.cat([atts_chatglm, prompt_attention_mask, end_attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.chatglm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        # output_text = self.llama_tokenizer.decode(outputs.logits[0][self.num_query_token:].argmax(1))
        output_text = self.chatglm_tokenizer.batch_decode(outputs.logits[:, self.num_query_token:].argmax(2), 
                                                        skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # output_text 应该是一个长度等于 batch 的list
        return {"loss": loss, "output_text": output_text}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.95,
        max_sentences=2,
        no_repeat_ngram_size=6,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.7,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - cloud (dict) : A dictionary containing the following keys:
                    - coord (Tensor): The coordinates of the points in the point cloud. The shape is [B, N, 3].
                    - color (Tensor): The colors of the points in the point cloud. The shape is [B, N, 3].
                text_input (str): prompt text, it will be used in each cloud in the batch
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
            max_sentences(int): The maximum number of sentences to be generated.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["cloud"]
        device = image["coord"].device
        with self.maybe_autocast():
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)

            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.chatglm_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

            if "text_input" in samples.keys():
                prompt = samples["text_input"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image_embeds.size(0)

            llama_tokens = self.chatglm_tokenizer(prompt, return_tensors="pt").to(device)
            input_embeds = self.chatglm_model.model.embed_tokens(llama_tokens.input_ids)
            input_embeds = torch.cat([inputs_opt, input_embeds], dim=1)         # 把 learnable query 和 prompt embedding 结果拼接起来
            attention_mask = torch.cat([atts_opt, llama_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = input_embeds.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = input_embeds.repeat_interleave(num_beams, dim=0)


            outputs = self.chatglm_model.generate(
                # input_ids=input_ids,
                inputs_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = llama_tokens.input_ids.shape[1]
            output_text = self.chatglm_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip().replace(" ", "") for text in output_text]
            # output_text = self.postprocess_text(output_text, device = device)
            return output_text


    @torch.no_grad()
    def generate_with_hidden_prompt(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.95,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.7,
    ):
        image = samples["cloud"]
        device = image["coord"].device
        with self.maybe_autocast():
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)

            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
    
            # Q-Former 的 learnable query 映射到 llama 的特征空间后得到的特征以及 attention mask
            inputs_query = self.chatglm_proj(query_output.last_hidden_state)
            atts_query = torch.ones(inputs_query.size()[:-1], dtype=torch.long).to(device)

            if "text_input" in samples.keys():
                prompt = samples["text_input"]
            else:
                prompt = self.prompt

            end_text = [prompt] * image_embeds.shape[0]
            end_token = self.chatglm_tokenizer(end_text, return_tensors="pt", padding="longest",
                                            truncation=True, max_length=self.max_txt_len).to(device)
            # llama_tokenizer 会自动在输入的文本前面加上 bos token, 所以这里要减去1
            end_attention_mask = end_token.attention_mask[:, 1:]
            end_token.input_ids = end_token.input_ids[:, 1:]
            end_embeds = self.chatglm_model.model.embed_tokens(end_token.input_ids)

            # 最后进行拼接  query + prompt
            input_embeds = torch.cat([inputs_query, end_embeds], dim=1)
            attention_mask = torch.cat([atts_query, end_attention_mask], dim=1)

            if use_nucleus_sampling:
                input_embeds = input_embeds.repeat_interleave(num_captions, dim=0)
                attention_mask = attention_mask.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                input_embeds = input_embeds.repeat_interleave(num_beams, dim=0)
                attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

            outputs = self.chatglm_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            output_text = self.chatglm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            output_text = [text.strip() for text in output_text]
            
            # output_text = self.postprocess_text(output_text, device = device)
            return output_text
        

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 32)

        pritrained_chatglm_model_path = cfg.get("pretrained_chatglm_path", "")
        qformer_encoder_layer = cfg.get("qformer_encoder_layer", 12)
        point_cloud_encoder_model = cfg.get("point_cloud_encoder_model", "")
        point_cloud_encoder_pretrain_model_path = cfg.get("point_cloud_encoder_model_path", None)
        freeze_point_cloud_encoder = cfg.get("freeze_cloud_encoder", True)

        model = cls(
            point_cloud_encoder_model = point_cloud_encoder_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            freeze_point_cloud_encoder = freeze_point_cloud_encoder,
            num_query_token=num_query_token,
            chatglm_model_path=pritrained_chatglm_model_path,
            prompt=prompt,
            max_txt_len=max_txt_len,
            qformer_encoder_layer=qformer_encoder_layer,
            point_cloud_encoder_pretrain_model_path = point_cloud_encoder_pretrain_model_path,
        )

        load_finetuned = cfg.get("load_finetuned", False)
        load_pretrained = cfg.get("load_pretrained", False)
        if(load_finetuned):
            logging.info("load fintuned blip2_llama model from {}".format(cfg.get("finetuned", None)))
            model.load_checkpoint_from_config(cfg)
            model.set_model_path(cfg.get("finetuned", None))
        elif(load_pretrained):
            logging.info("load pretrained blip2_llama model from {}".format(cfg.get("pretrained", None)))
            model.load_checkpoint_from_config(cfg)
            model.set_model_path(cfg.get("pretrained", None))

        return model
