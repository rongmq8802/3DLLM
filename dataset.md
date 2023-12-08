# 数据集合说明
集群：
数据集路径：`/public/public_data/3DLLM/`

## S3dis
集群：
point cloud: `/public/public_data/3DLLM/S3DIS/`

text: `/public/public_data/3DLLM/s3dis_text/`

english + 房间类型: `"/public/public_data/3DLLM/s3dis_text/s3dis_text.json"`

中文+ 房间类型：`"/public/public_data/3DLLM/s3dis_text/ZN-s3dis_text.json"`

## scannet

point cloud: 

`/public/public_data/3DLLM/Scannet_original/ply/`

`/public/public_data/3DLLM/Scannet_original/scannetply_train/`

`/public/public_data/3DLLM/Scannet_original/scannetply_val/`

text: 

英文：`/public/public_data/3DLLM/scannet_text/scannet_text.json`

中文：`/public/public_data/3DLLM/scannet_text/ZN-scannet_text.json`

  
  

## struture3d

point cloud:`/public/public_data/3DLLM/str3d_pc/pc/` including train/test/val

text: `/public/public_data/3DLLM/str3d_text/` including train/test/val

## 合并后的：

point cloud：

train:`/public/public_data/3DLLM/merge/train/`

test:`"/public/public_data/3DLLM/merge/val/`

text：

`/public/public_data/3DLLM/merge/ZN-merge.json`


## 已完成

1.structured3d 全景图片转换成ply点云

2.构造layout数据集，示例：

There are 10 rooms, incluing 1 living room, 1 kitchen, 3 bedrooms, 2 bathrooms, 1 balcony, 1 undefined room, and there are 8 doors,and there are 10 windows,bedroom and living room have contections.living room has contections to the outside.living room and bedroom have contections.bedroom and living room have contections.living room and bathroom have contections.balcony and living room have contections.living room and kitchen have contections.bedroom and bathroom have contections.living room and balcony have contections.living room has 1 window, kitchen has 2 windows, bedroom has 3 windows, bathroom has 2 windows, balcony has 1 window, undefined has 1 window, 


3.在S3dis数据集添加房间类型描述：

A room with a table, chairs, and a clock on the wall. a room filled with desks and chairs with books on them. a classroom with a desk with a computer and chairs. a row of chairs sitting in front of a window. a white wall with a picture of a man on it. **this is a conferenceRoom**.

4.完成英文数据集至中文数据集的翻译

实例：

s3dis:

"Area_1/WC_1": "带水槽、马桶和淋浴间的浴室。有两个水槽和一面镜子的浴室。厨房配有冰箱和微波炉。有厕所、水槽和门的走廊。洗手间的门是开着的，墙上挂着一个钟。建筑物墙上的标志。这是一个厕所。"

scannet:

"scene0000_00": "照片显示，一只猫躺在房间的床上，一辆自行车停在门前，一个吉他盒放在地毯旁边的地板上。其他图片显示厨房有一个水槽，浴室有两个水槽和一把椅子，客厅有一张沙发和一台电视，卧室有一张床和一张桌子。"

str3d:
"  "/data3/rmq/scene_00003.ply": "有7个房间，包括1间客厅，1间厨房，2间卧室，1间浴室，1个阳台，1个书房，有7个门，有5个窗户，浴室和客厅有门连通。书房和客厅有门连通。阳台和客厅有门连通。客厅和厨房有门连通。卧室和客厅都有门连通。客厅和卧室都有连接。客厅向外连接。厨房有1个窗户，卧室有2个窗户。"

5.合并s3dis和scannet数据集

6.处理得到scannet点云pth数据

## 未完成
