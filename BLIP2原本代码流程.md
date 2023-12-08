# 程序整体流程

程序调用代码为 `python train.py --cfg-path --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml`

在`train.py` 中，首先解析命令行参数，得到配置文件的路径，然后读取配置文件。

## 配置文件解析部分

配置文件整体上可以分为三个部分：`model` 定义模型基础信息，`datasets` 定义数据集，`run` 定义训练、验证时的各种参数，例如epoch数量，学习率等等

首先读取命令行上的其他参数，这些参数的优先级最高，会覆盖配置文件里同名的配置参数。然后分别解析配置文件里的三个部分的配置。`run` 部分最简单，直接读取即可，没有修改。

`model` 部分的解析是得到了模型的名字(`blip2`)以及模型的类型(`pretain`)，根据这二者得到了默认的模型配置文件，即 

```python
model_config_path = model_cls.default_config_path(model_type=model_type)  # model_cls=blip2  model_type=pretain
```

在 `blip2_qformer.py` 文件中，定义了三种模型的配置文件的路径

```python
 PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }
```

从 `pretrain` 对应的配置文件中读取模型的配置参数，这些参数和我们在命令行里提供的模型配置参数相结合，得到最终的模型配置参数

 `datasets` 部分的解析是根据配置得到数据集的名字，分别是 `coco_caption` 对应于 `class COCOCapBuilder(BaseDatasetBuilder)` 以及 `vg_caption` 对应于 `class VGCaptionBuilder(BaseDatasetBuilder)` 。

以 `coco_caption` 为例，根据它的名字得到数据集的类型是  `class COCOCapBuilder(BaseDatasetBuilder)` ，然后得到这个数据集的默认配置文件路径为 `configs/datasets/coco/defaults_cap.yaml`，从中读取相关的配置，最后和我们提供的配置相融合，得到最终的数据集配置参数。数据集整体的配置形式如下

```shell
datasets:     coco_caption :  coco_caption 默认配置
							  提供的配置
							  命令行配置
			   vg_caption:    vg_caption 默认配置
			   			      提供的配置
			   			      命令行配置
```



## 准备训练

在训练开始之前，还要做一些准备工作，包括初始化分布式训练 `init_distributed_mode(cfg.run_cfg)`，设置随机种子 `setup_seeds(cfg)`，设置logger `setup_logger()`，打印关键的变量信息 `cfg.pretty_print()`。

设置整体的任务 `task = tasks.setup_task(cfg)` ，也就是什么类型的任务，可以是视觉问答，文本描述，分类等等，这里使用的是 `ImageTextPretrainTask`，最终实际的调用就是 `class BaseTask`

设置加载的数据集 `datasets = task.build_datasets(cfg)`。根据提供的配置文件，数据集共有两个，对应于两个数据集的 Builder。在Builder中，会根据配置文件建立不同的数据集(train / test / val) ，然后对文字和图像分别通过 `text_processor` 和 `vis_processor` 进行一些预处理。`processer` 也有很多不同的类型，每个类型都有自己的名字，具体使用哪种是在配置文件中的 `processor.name` 指定的。在 `vis_processor` 中会对图像进行一些裁剪、翻转、颜色变换等数据增强操作。在 `text_processor` 中，会先把文字中的特殊符号（如句号、感叹号、引号、括号、冒号等）变成一个空格代替，然后把多个连续的空格变成一个空格，接着去除这句话开头和结尾的空格以及换行符，最后会对文字截取，如果文字长度超过了限制(max_words=50)那就只保留前N个字。

使用 COCO 数据集时，如果是训练，最终调用的Dataset是 `CaptionDataset`，那么得到的数据形式为 `{"image": image, "text_input": caption, "image_id": self.img_ids[ann["image_id"]]}`

使用COCO的数据集时，如果是测试，最终调用的 Dataset 是 `COCOCapEvalDataset` ，得到的数据形式为 `{"image": image, "image_id": img_id, "instance_id": ann["instance_id"]}`

使用VG的数据集时，只能支持训练，最终调用的 Dataset 是 `ImageTextPairDataset`，得到的数据形式为 `{"image": image, "text_input": caption}`

```shell
coco_caption :  train:  {"image": image, "text_input": caption, "image_id": self.img_ids[ann["image_id"]]}
				val:  {"image": image, "image_id": img_id, "instance_id": ann["instance_id"]}
				test:  {"image": image, "image_id": img_id, "instance_id": ann["instance_id"]}

vg_caption:    train:  {"image": image, "text_input": caption}
```

设置模型 `model = task.build_model(cfg)` 在这一部分里，首先会根据 `cfg.model.model_arch` 得到模型的具体名字，即 `blip2`，然后根据其他相关的配置创建一个 `blip2` 模型，具体代码如下

```python
def from_config(cls, cfg):		# cls 是 class Blip2Qformer(Blip2Base)  cfg 是 config.model 即所有配置中关于model的那部分
        vit_model = cfg.get("vit_model", "eva_clip_g")		# 从中获取 vit_model，如果没有相关配置就使用默认参数 eva_clip_g
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,						# eva_clip_g
            img_size=img_size,							# 224
            drop_path_rate=drop_path_rate,				# 0.0
            use_grad_checkpoint=use_grad_checkpoint,	# False
            vit_precision=vit_precision,				# fp16
            freeze_vit=freeze_vit,						# True
            num_query_token=num_query_token,			# 32
            cross_attention_freq=cross_attention_freq,	# 2
            max_txt_len=max_txt_len,					# 32
        )
        model.load_checkpoint_from_config(cfg)	# 最终结果是从 configs/models/blip2/blip2_pretrain.yaml 文件中 pretrained 路径读取

        return model
```

设置运行的整体 `runner`，这里最终使用的runner 类型是 `class RunnerBase`

## 开始训练

开始训练时调用的顺序是 `runner.train() -> runner.train_epoch() -> runner.task.train_epoch() -> runner.task._train_inner_loop()`

在 `_train_inner_loop()` 中，会首先定义logger，然后从 `data_loader` 中读取数据，并把数据送到显卡上，最后把数据送入 `model` 中并返回loss

```python
def train_step(self, model, samples):
    loss = model(samples)["loss"]
    return loss

def _train_inner_loop():
    ......
    ......
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
        loss = self.train_step(model=model, samples=samples)
```

# Q-Former 整体架构

Q-Former 的结构定义在 `class Blip2Qformer(Blip2Base)`中，整个模型的初始化是在 `model = task.build_model(cfg)` 这行代码中

## 初始化

### 初始化 Tokenizer

在Q-Former的初始化函数 `__init__()` 中，主要初始化了 `tokenizer` 。这里使用了 BERT 作为分词器，并在其中加入了名为`bos_token`的特殊标记，其值为`[DEC]` 。这里使用的 tokenizer 是 `bert-base-uncased`

```python
def init_tokenizer(cls):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer
```

### 初始化 vision encoder

然后初始化视觉编码器 (`vision_encoder`)，即 `self.visual_encoder, self.ln_vision = self.init_vision_encoder()`

根据配置文件，这里使用的是 `eva_clip_g` 模型，这是一个 `ViT-G` 模型，预训练结果存储在 `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth` 中，由于预训练模型和当前希望使用的模型在位置嵌入方面不同，所以需要对预训练模型的位置嵌入层进行插值，使它的位置嵌入满足当前模型的需要，即 `interpolate_pose_embed`。如果使用半精度训练，那么还需要把所有权重都改为半精度。相关代码为

```python
url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
cached_file = download_cached_file(url, check_hash=False, progress=True)
state_dict = torch.load(cached_file, map_location="cpu")    
interpolate_pos_embed(model,state_dict)
model.load_state_dict(state_dict, strict=False)
if precision == "fp16":
    convert_weights_to_fp16(model)
```

之后会定义一个视觉的 LayerNorm，即 `ln_vision = LayerNorm(visual_encoder.num_features)`

由于设置了 `freeze_vit` 即 ViT 是固定的，不被训练的，因此要把视觉编码器中的所有的参数的 `require_grad` 设置为 `False`

### 初始化Q-Former

首先从 BERT 中读取基础的配置，然后对这些配置进行一些修改，得到Q-Former使用的配置，接着根据这个配置进行Q-Former的初始化。初始化过程主要是根据BERT模型初始化一个Q-Former模型，然后建立 32个 query tokens

```python
def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.encoder_width = vision_width		# eva_vit_g.embed_dim = 1408
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq	# 2
    encoder_config.query_length = num_query_token				# 32
    Qformer = BertLMHeadModel.from_pretrained(
        "bert-base-uncased", config=encoder_config
    )
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)		# hidden_size=768
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens
```

这里的 `BertLMHeadModel.from_pretained()` 做了两件事，第一是初始化了一个 BERT 模型，第二是初始化了一个BERT分类头 `BertOnlyMLMHead()` 最后对所有的模型权重初始化。

```shell
transformers.modeling_utils.PreTrainedModel
					|
					|-----BertPreTrainedModel
					|			|------BertModel
					|			|------BertLMHeadModel
					|-----
```

得到Q-Former后，要对它的词嵌入矩阵进行一下resize，这是因为它的词嵌入矩阵是来自BERT的，但目前使用的BERT Tokenizer中额外加入了`[DEC]` 标志，为了让词嵌入矩阵还要对其中的一些参数进行权重共享，也就是把原本模型中的 `xxxx_query` 和 `xxxx` 建立关联，具体代码如下

```python
state_dict = self.Qformer.state_dict()
    for name, param in self.Qformer.named_parameters():
        if "_query" in name:
            key_orig = name.replace("_query", "")
            param.data.copy_(state_dict[key_orig])
```

最后会初始化一些线性层，对特征的维度进行变换

```python
self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)	# hidden_size=768  embed_dim=256
self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
```

## 前向传播











