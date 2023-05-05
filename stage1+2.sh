cd /public/home/mengqi_rong8/Code/3DLLM/


job_id="cap_and_structure_v2"
node_stage1=5
node_stage2=2
master_port=28880

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

python -m torch.distributed.run \
        --nproc_per_node=${node_stage1} \
        --master_port=${master_port} \
        train_3d.py \
        --cfg-path /public/home/mengqi_rong8/Code/3DLLM/lavis/projects/blip2_3d/train/pretrain_stage1.yaml \
        --job_id ${job_id} 





python -m torch.distributed.run \
        --nproc_per_node=${node_stage2} \
        --master_port=${master_port} \
        train_3d.py \
        --cfg-path /public/home/mengqi_rong8/Code/3DLLM/lavis/projects/blip2_3d/train/pretrain_stage2.yaml \
        --job_id ${job_id} \
        --options model.load_pretrained=True \
        model.pretrained=/public/home/mengqi_rong8/Code/3DLLM/lavis/output/BLIP2/Pretrain_stage1/${job_id}/checkpoint_best.pth


OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5,6,7 \
python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=28882 \
        train_3d.py \
        --cfg-path /public/home/mengqi_rong8/Code/3DLLM/lavis/projects/blip2_3d/train/pretrain_stage2.yaml \
        --job_id cap_and_structure_v3 \
        --options model.load_pretrained=True \
        model.pretrained=/public/home/mengqi_rong8/Code/3DLLM/lavis/output/BLIP2/Pretrain_stage2/cap_and_structure_v2/checkpoint_38.pth