import os
import sys
import wandb
import random
import numpy as np

import torch
import torch.distributed as dist

from dataset import NuScenesDatasetWrapper


def get_run_name(args):

    num_steps = str(args.num_steps//1000) + 'k'

    line_reconstruction_weight = str(args.line_reconstruction_weight)
    box_reconstruction_weight = str(args.box_reconstruction_weight)
    line_ce_weight = str(args.line_ce_weight)
    box_ce_weight = str(args.box_ce_weight)
  
    res = str(args.bev_resolution)

    run_name = res + '_lr' + str(args.learning_rate) + '_bs' + str(args.batch_size) + '_steps' + num_steps + '_line' + line_reconstruction_weight + '_box' + box_reconstruction_weight + '_linece' + line_ce_weight + '_boxce' + box_ce_weight + '_normbycnt_angle'
    return run_name


# === ================ ===
# === Model Related ===

def init_model(args):

    model_name = args.model_name
    
    if model_name == 'bev-autoencoder':
        from models.wrapper import RVAEWrapper
        model = RVAEWrapper(args)
    else:
        raise NotImplementedError
    
    return model.cuda()


def print_model_summary(args, model):

    if args.model_name == 'simplebev':
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6
        encoder_params = sum(p.numel() for p in model.model.encoder.parameters() if p.requires_grad) // 1e6
        bev_compressor_params = sum(p.numel() for p in model.model.bev_compressor.parameters() if p.requires_grad) // 1e6
        decoder_params = sum(p.numel() for p in model.model.decoder.parameters() if p.requires_grad) // 1e6
        print('#########################################')
        print(f'Number of parameters (All): {int(model_params)}M')
        print('#########################################')
        print('#########################################')
        print('#########################################')
        print(f'Number of parameters (Encoder): {int(encoder_params)}M')
        print(f'Number of parameters (Head): {int(bev_compressor_params + decoder_params)}M')
        print(f'     # parameters (BEV Compressor): {int(bev_compressor_params)}M')
        print(f'     # parameters (Decoder): {int(decoder_params)}M')
        print('#########################################')

    else:
        pass


# === ================ ===
# === Logger Related ===

def init_logger(args,run_name):

    project_name = args.project
    # if args.validate:
    #     project_name += '_val'
    #     run_name = args.checkpoint_path.split('/')[-2]

    wandb.init(
                project=project_name, 
                name=run_name
                ) 



# === ================ ===
# === Training Related ===

def restart_from_checkpoint(args, run_variables, **kwargs):

    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None and checkpoint[key] is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


# === ================ ===
# === Data Related ===

def get_datasets(args):
    
    datamodule = NuScenesDatasetWrapper(args)

    trainset = datamodule.train()
    valset = datamodule.val()

    print('#########################################')
    print('Trainset size : ', len(trainset))
    print('Valset size   : ', len(valset))
    print('#########################################')

    return trainset, valset


def worker_rnd_init(x):
    np.random.seed(13 + x)


import torch

def rvae_collate_fn(batch):

    features_list = []
    city_token_list = []
    color_mask_list = []

    lane_vector_list = []
    lane_mask_list = []

    vehicles_vector_list = []
    vehicles_mask_list = []

    # 1. Collect all data from the batch
    for item in batch:
        features_list.append(item['features'])  # shape (Z, X, 4)
        # city_token_list.append(item['city_token'])
        color_mask_list.append(item['color_mask'])

        lane_vector_list.append(item['targets']['LANES']['vector'])   # shape (L, 20, 2)?
        lane_mask_list.append(item['targets']['LANES']['mask'])

        vehicles_vector_list.append(item['targets']['VEHICLES']['vector'])  # shape (V, 5)?
        vehicles_mask_list.append(item['targets']['VEHICLES']['mask'])

    features_batch = torch.stack(features_list, dim=0)  # (B, Z, X, 4) 
    color_mask_batch = torch.stack(color_mask_list, dim=0)  # (B, 3, X)
    lane_vector_batch = torch.stack(lane_vector_list, dim=0)        # (B, L, 20, 2)
    lane_mask_batch = torch.stack(lane_mask_list, dim=0)            # e.g. (B, L) or (B, L, 20)

    vehicles_vector_batch = torch.stack(vehicles_vector_list, dim=0)   # (B, V, 5)
    vehicles_mask_batch = torch.stack(vehicles_mask_list, dim=0)       # e.g. (B, V)

    collated_batch = {
        'features': features_batch,               # (B, Z, X, 4)
        'color_mask': color_mask_batch,           # (B, 3, X)
        'targets': {
            'LANES': {
                'vector': lane_vector_batch,      # (B, L, 20, 2)
                'mask': lane_mask_batch
            },
            'VEHICLES': {
                'vector': vehicles_vector_batch,  # (B, V, 5)
                'mask': vehicles_mask_batch
            }
        }
    }

    return collated_batch



def get_dataloaders(args, trainset, valset):

    train_sampler = torch.utils.data.DistributedSampler(trainset, num_replicas=args.gpus, rank=args.gpu, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.batch_size // args.gpus,
        num_workers=6,                                  # cpu per gpu
        worker_init_fn=worker_rnd_init,
        drop_last=True,
        pin_memory=False,
        collate_fn=rvae_collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=1,
        shuffle=False, 
        num_workers=6, 
        drop_last=False, 
        pin_memory=False,
        collate_fn=rvae_collate_fn
        )
     
    return train_dataloader, val_dataloader



def get_random_subset_dataloader(dataset, n=100):
    
    datasubset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), n))
    dataloader = torch.utils.data.DataLoader(datasubset, 
        batch_size=1,
        shuffle=False, 
        num_workers=6, 
        drop_last=False, 
        pin_memory=True)
    
    return dataloader



def get_scheduler(args, optimizer, T_max=None):

    if T_max is None:
        T_max = args.num_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, T_max+100,
        pct_start=0.0, cycle_momentum=True, anneal_strategy='cos', div_factor=10.0, final_div_factor=10,
        base_momentum=0.85, max_momentum=0.95)
    
    return scheduler


# === ===================== ===
# ===  Distributed Settings ===

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L467C1-L499C42

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = args.gpus
        args.gpu = args.rank % torch.cuda.device_count()

    # launched naively with `python train.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, 'env://'), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)