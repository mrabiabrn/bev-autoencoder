torchrun --master_port=12396 --nproc_per_node=4 train.py --config configs/base.yaml

# NCCL_P2P_DISABLE=1 WANDB_MODE=offline t