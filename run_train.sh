NCCL_P2P_DISABLE=1, torchrun --master_port=10390 --nproc_per_node=4 train.py --config configs/base.yaml

# NCCL_P2P_DISABLE=1 WANDB_MODE=offline t

torchrun --master_port=10393 --nproc_per_node=1 train.py --config configs/base.yaml