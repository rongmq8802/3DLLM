目前的代码进行了以下修改：  
1. `lavis/models/blip2_models` 是使用三维点云作为输入的代码，原本的 blip2 代码被放到了同级目录的 `blip2_models_reserve` 中，理论上老代码应该不会运行
2. `lavis/projects/blip2_3d/pretrain_stage1.yaml` 是当前整体的配置 
3. `lavis/configs/models/blip2_3d/blip2_point_cloud.yaml` 是当前的Q-former的配置
4. `lavis/configs/datasets/point_cloud/defaults_cap.yaml` 是当前数据集使用的配置
5. `lavis/processors/cloud_processors.py` 实现了相关的对点云和中文caption进行预处理的代码
6. `lavis/tasks/pointcloud_text_pretrain.py` 实现了用于当前任务的task, 实际上就是和BaseTask一样
7. `lavis/datasets/datasets/cloud_text_pair_datasets.py` 实现了点云+caption的数据集本身的定义, 包括如何读取数据
8. `lavis/datasets/builders/cloud_text_pair_builder.py` 实现了构建相关数据集的方法, 包括读取数据集的配置文件, 设置数据的预处理
9. ``


TODO:
1. 数据集构造
builder = registry.get_builder_class(name)(dataset_config)  
dataset = builder.build_datasets()
注册processor用于预处理数据，包括cloud_processor 和中文的 text_processor
建立一个假数据测试
2. Point Transformer 构建 
3. llama 模型









RongMengqi 修改
1. 上传了PointTransformer网络，并打开了一个接口，可以控制输出的特征点的数目
2. 完成了load_pairs函数（从文件中读取点云模型的路径和对应的文本描述）cloud_text_pair_datasets.py
3. 完成了对点云的预处理代码撰写 主要是cloud_processors.py



TODO:
1. models/libs/pointops下面的setup可能需要重新安装