# 1.采用的3D Backbone为发表在NIPS2022的Point Transformer网络结构 



github链接 https://github.com/Gofinge/PointTransformerV2 和 https://github.com/Pointcept/Pointcept

其中涉及到SPVCNN、SparseUNet、PointGroup部分的环境不用装

```python
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..



# Open3D (Visualization)
pip install open3d

# PPT
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# stratified transformer
pip install torch-points3d
# fix dependence, caused by install torch-points3d 
pip uninstall SharedArray
pip install SharedArray==3.2.1

cd libs/pointops2
python setup.py install
cd ../..

```





==安装过程中我们遇到过的坑==，python setup.py install 这一步在安装时没问题，但是运行程序的时候会报错（libstdc++.so.6: version `GLIBCXX_3.4.26' not found），可能是GCC版本的问题（gcc --version）。GCC的版本不能太低（所内A100集群的是4.8就不太行），我们在集群上conda环境里重新安装过7.3是可以的，再高的版本应该也没问题（我们自己的服务器是11.2也是可以的）。