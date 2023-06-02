# 3DLLM

## 环境配置

配置基本环境, 包含BLIP2需要的 pytorch, transformers 等环境

```
conda env create environment.yaml
```

配置PointTransformer环境，这是三维特征提取所需要的

具体参见 [PointTransformer环境配置.md](PointTransformer环境配置.md)

## 工具代码

### :cat: llama原版weight转huggin face格式

#### 创建环境

```bash
pip install transformers
pip install accelerate
pip install protobuf==3.20.0
```

#### 使用

```bash
python transform.py --input_dir "llama_weight_dir" --model_size "X"B --output_dir "output_dir"
```

# 运行代码

运行代码具体见 [代码运行方式.md](代码运行方式.md)