# 1.第一步，需要修改blip2.py的第70行，即点云模型的encoder定义部分

## 1) 对于第一次训练的小模型S3DIS，网络结构如下**

```python
            cloud_encoder = PointTransformerV2(in_channels=6,
                                             num_classes=13,
                                             patch_embed_depth=2,
                                             patch_embed_channels=48,
                                             patch_embed_groups=6,
                                             patch_embed_neighbours=16,
                                             enc_depths=(2, 6, 2),
                                             enc_channels=(96, 192, 384),
                                             enc_groups=(12, 24, 48),
                                             enc_neighbours=(16, 16, 16),
                                             dec_depths=(1, 1, 1),
                                             dec_channels=(48, 96, 192),
                                             dec_groups=(6, 12, 24),
                                             dec_neighbours=(16, 16, 16),
                                             grid_sizes=(0.1, 0.2, 0.4),
                                             attn_qkv_bias=True,
                                             pe_multiplier=False,
                                             pe_bias=True,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.3,
                                             enable_checkpoint=False,
                                             unpool_backend="interp",
                                             num_features=256,
                                             checkpoint_path=pretrained_model_path,)
```

## 2) 对于第二次训练的模型scannet，甚至以后其他更多的数据，都采用如下网络结构（但是需注意sacannet是20个语义类别，其他的数据集类别可能不一样，有时候需要修改一下其中的num_classes）：

```python
            cloud_encoder = PointTransformerV2(in_channels=6,
                                             num_classes=20,
                                             patch_embed_depth=1,
                                             patch_embed_channels=48,
                                             patch_embed_groups=6,
                                             patch_embed_neighbours=8,
                                             enc_depths=(2, 2, 6, 2),
                                             enc_channels=(96, 192, 384, 512),
                                             enc_groups=(12, 24, 48, 64),
                                             enc_neighbours=(16, 16, 16, 16),
                                             dec_depths=(1, 1, 1, 1),
                                             dec_channels=(48, 96, 192, 384),
                                             dec_groups=(6, 12, 24, 48),
                                             dec_neighbours=(16, 16, 16, 16),
                                             grid_sizes=(0.06, 0.15, 0.375, 0.9375),
                                             attn_qkv_bias=True,
                                             pe_multiplier=False,
                                             pe_bias=True,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.3,
                                             enable_checkpoint=False,
                                             unpool_backend="map",
                                             num_features=256,
                                             checkpoint_path=pretrained_model_path,)
```



# 2.第二步，scannet点云的预处理可以修改一下，即cloud_processors.py，大致修改为如下情况

## 1）对于训练集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.2)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1/64, 1/64], axis='x', p=0.5)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1/64, 1/64], axis='y', p=0.5)(pcd_dict)
        pcd_dict = RandomScale(scale=[0.9, 1.1])(pcd_dict)
        pcd_dict = RandomFlip(p=0.5)(pcd_dict)
        pcd_dict = RandomJitter(sigma=0.005, clip=0.02)(pcd_dict)
        pcd_dict = ElasticDistortion(distortion_params=[[0.2, 0.4], [0.8, 1.6]])(pcd_dict)
        pcd_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(pcd_dict)
        pcd_dict = ChromaticTranslation(p=0.95, ratio=0.05)(pcd_dict)
        pcd_dict = ChromaticJitter(p=0.95, std=0.05)(pcd_dict)        
        # 随机下采样是我自己实现的，而对于其他的数据变换方式是原来的PointTransformer2的代码
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```

## 2）对于测试集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```


# 3.第三步，有几种数据增强是之前的代码中未涉及到的，现补充(直接粘贴到cloud_processors.py的开头部分即可)

## 1）RandomDrop
```python
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
            upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict
```

## 2）RandomRotate
```python
class RandomRotate(object):
    def __init__(self,
                 angle=None,
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

```

## 3）ElasticDistortion
```python
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(data_dict["coord"], granularity, magnitude)
        return data_dict
```

## RandomSample函数更新
```python
class RandomSample(object):
    def __init__(self, max_num = 80000):
        self.max_num = max_num
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            if data_dict["coord"].shape[0] > self.max_num:
                choice = np.random.choice(data_dict["coord"].shape[0], self.max_num, replace=False)
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][choice]
            else:
                #若不够这么多点，则随机的重复采样
                choice = np.random.choice(data_dict["coord"].shape[0], self.max_num - data_dict["coord"].shape[0], replace=True)
                for key in data_dict.keys():
                    data_dict[key] = np.concatenate([data_dict[key], data_dict[key][choice]], axis=0)
        return data_dict
```

# 4.以前的对于S3DIS数据集的预处理方式（暂存一下）

## 1）对于训练集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = RandomScale(scale=[0.9, 1.1])(pcd_dict)
        pcd_dict = RandomFlip(p=0.5)(pcd_dict)
        pcd_dict = RandomJitter(sigma=0.005, clip=0.02)(pcd_dict)
        pcd_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(pcd_dict)
        pcd_dict = ChromaticTranslation(p=0.95, ratio=0.05)(pcd_dict)
        pcd_dict = ChromaticJitter(p=0.95, std=0.05)(pcd_dict)
        # pcd_dict = Voxelize(voxel_size=0.04, hash_type='fnv', mode='train',
        #             keys=("coord", "color", "label"), return_discrete_coord=True)(pcd_dict)
        # pcd_dict = SphereCrop(point_max=100000, mode='random')(pcd_dict)

        # 随机下采样是我自己实现的，而对于其他的数据变换方式是原来的PointTransformer2的代码
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```

## 2）对于测试集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = Copy(keys_dict={"coord": "origin_coord", "label": "origin_label"})(pcd_dict)
        # pcd_dict = Voxelize(voxel_size=0.04, hash_type='fnv', mode='train',
        #             keys=("coord", "color", "label"), return_discrete_coord=True)(pcd_dict)
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```



# 5.把模型的encoder和seg_head不加载
## 1) 可以去掉模型中的num_classes这一类别以及decoder这一部分，以后对于不同的数据集就不再需要修改。此外，还修改了预训练模型参数的加载，新增若在new_model_state_dict中，但是不在预训练的模型参数中就报错一步骤。在本地机中测试通过
```python
class PointTransformerV2(nn.Module):
    def __init__(self,
                 in_channels,
                #  num_classes,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2, 2, 6, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                #  dec_depths=(1, 1, 1, 1),
                #  dec_channels=(48, 96, 192, 384),
                #  dec_groups=(6, 12, 24, 48),
                #  dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map",
                 num_features=256,
                 checkpoint_path=None,
                 ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.num_features = num_features
        self.checkpoint_path = checkpoint_path
        # assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        # assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        # assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        # assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        # dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        # dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        # self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            # dec = Decoder(
            #     depth=dec_depths[i],
            #     in_channels=dec_channels[i + 1],
            #     skip_channels=enc_channels[i],
            #     embed_channels=dec_channels[i],
            #     groups=dec_groups[i],
            #     neighbours=dec_neighbours[i],
            #     qkv_bias=attn_qkv_bias,
            #     pe_multiplier=pe_multiplier,
            #     pe_bias=pe_bias,
            #     attn_drop_rate=attn_drop_rate,
            #     drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
            #     enable_checkpoint=enable_checkpoint,
            #     unpool_backend=unpool_backend
            # )
            self.enc_stages.append(enc)
            # self.dec_stages.append(dec)
        # self.seg_head = nn.Sequential(
        #     nn.Linear(dec_channels[0], dec_channels[0]),
        #     PointBatchNorm(dec_channels[0]),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dec_channels[0], num_classes)
        # ) if num_classes > 0 else nn.Identity()

        if self.checkpoint_path is not None:
            self.load_pretrained_model()


    def forward(self, data_dict):
        # 按照默认的DataLoader读入数据，是存在batch这种信息的，而PointTransformer在读入时会把所有batch的信息拼在一起
        # 所以在这里要做一个特别的处理，把batch的信息合并到一个torch.Tensor里面

        data_dict = self.merge_batch(data_dict)

        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        # for i in reversed(range(self.num_stages)):
        #     skip_points, cluster = skips.pop(-1)
        #     points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points
        # seg_logits = self.seg_head(feat)
        # feat = points[1]

        # 按照offset的划分，将feat分成多个部分
        feat_list = []
        for i in range(offset.shape[0]):
            if i == 0:
                featurelength = offset[i]
                # 如果特征点的数目太多的话，则从其中随机的选出N个点
                if featurelength > self.num_features:
                    random_sample = random.sample(range(featurelength), self.num_features)
                    feat_list.append(feat[random_sample, :])
                    # feat_list.append(feat[:offset[i], :])
                else:
                    #若不够self.num_features个维度，则在填充了原来的数据的情况下，再随机采点
                    expand_dim = self.num_features - featurelength
                    # random_sample = random.sample(range(featurelength), expand_dim)
                    random_sample = torch.randint(featurelength, (expand_dim,))
                    #将已有的特征点和随机采样的特征点拼接在一起
                    feat_list.append(torch.cat((feat[:offset[i],:],feat[random_sample, :]), dim=0))
            else:
                featurelength = offset[i] - offset[i - 1]
                if featurelength > self.num_features:
                    random_sample = random.sample(range(featurelength), self.num_features)
                    random_sample = torch.tensor(random_sample)
                    feat_list.append(feat[random_sample + offset[i - 1].item(), :])
                    #feat_list.append(feat[offset[i - 1]:offset[i], :])
                else:
                    expand_dim = self.num_features - featurelength
                    # random_sample = random.sample(range(featurelength), expand_dim)
                    random_sample = torch.randint(featurelength, (expand_dim,))
                    # random_sample = torch.tensor(random_sample)
                    feat_list.append(torch.cat((feat[offset[i - 1]:offset[i],:],feat[random_sample + offset[i - 1].item(), :]), dim=0))
        
        #以现在的网络，最后的Encoder输出是N*C的特征，N是点云的数量，C是特征的维度
        feat_list = torch.stack(feat_list, dim=0)
        return feat_list

    
    # def load_pretrained_model(self):
    #     pretrained_dict = torch.load(self.checkpoint_path, map_location="cpu")
    #     model_dict = self.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.load_state_dict(model_dict)
    #     logging.info("Load pretrained PointTransformer model from {}".format(self.checkpoint_path)) 

    def load_pretrained_model(self):
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # print(self.state_dict().keys())
        #判断model的模型参数和预训练的模型参数是否完全一致
        for key in self.state_dict().keys():
            # 如果key不在预训练模型的参数中，并且key不是以seg_head和dec_stages开头的参数
            if key not in state_dict.keys() and not key.startswith("seg_head") and not key.startswith("dec_stages"):
                print("key {} is not in pretrained model".format(key))

        
        # 删除不需要的参数
        for key in list(state_dict.keys()):
            if key.startswith("seg_head") or key.startswith("dec_stages"):
                del state_dict[key]
        self.load_state_dict(state_dict)
        print("Load pretrained model from {}".format(self.checkpoint_path))


    def merge_batch(self, data_dict):
        # data_dict是一个字典，字典中的每一个元素都是一个torch.Tensor
        # 按照batch的维度对data_dict进行拼接
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].view(-1, *data_dict[key].shape[2:])

        # 给offset中的数，每个数都加上前面所有batch的点的数量
        if "offset" in data_dict.keys():
            offset = data_dict["offset"]
            offset = torch.cumsum(offset, dim=0)
            data_dict["offset"] = offset
        
        return data_dict
```


## 2）对于模型的初始化对于scannet以及其他的数据集，以后可以固定
```python
            cloud_encoder = PointTransformerV2(in_channels=6,
                                            #  num_classes=20,
                                             patch_embed_depth=1,
                                             patch_embed_channels=48,
                                             patch_embed_groups=6,
                                             patch_embed_neighbours=8,
                                             enc_depths=(2, 2, 6, 2),
                                             enc_channels=(96, 192, 384, 512),
                                             enc_groups=(12, 24, 48, 64),
                                             enc_neighbours=(16, 16, 16, 16),
                                            #  dec_depths=(1, 1, 1, 1),
                                            #  dec_channels=(48, 96, 192, 384),
                                            #  dec_groups=(6, 12, 24, 48),
                                            #  dec_neighbours=(16, 16, 16, 16),
                                             grid_sizes=(0.06, 0.15, 0.375, 0.9375),
                                             attn_qkv_bias=True,
                                             pe_multiplier=False,
                                             pe_bias=True,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.3,
                                             enable_checkpoint=False,
                                             unpool_backend="map",
                                             num_features=256,
                                             checkpoint_path=pretrained_model_path,)
```
