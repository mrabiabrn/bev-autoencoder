import yaml
import torch
import argparse


def print_args(args):

    print("====== Training ======")
    print(f"project name: {args.project}\n")

    print(f"model: {args.model_name}\n")

    print(f"resolution: {args.resolution}\n")

    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    print(f"effective_batch_size: {args.batch_size * args.gradient_acc_steps}")
    print(f"num_epochs: {args.num_epochs}")

    print("====== ======= ======\n")


def set_args(cfg):

    cfg.project  = 'RVAE'

    assert 0 <= cfg.threshold <= 1, "Config threshold must be in [0,1]"

    assert cfg.positional_embedding in ["learned", "sine"]
    assert cfg.activation in ["relu", "gelu", "glu"]

    cfg.pixel_frame = (int(cfg.frame[0] / cfg.pixel_size), int(cfg.frame[1] / cfg.pixel_size))

    cfg.latent_frame = (int(cfg.pixel_frame[0] / cfg.down_factor), int(cfg.pixel_frame[1] / cfg.down_factor))

    cfg.patches_frame = (int(cfg.latent_frame[0] / cfg.patch_size), int(cfg.latent_frame[1] / cfg.patch_size))

    cfg.num_patches = cfg.patches_frame[0] * cfg.patches_frame[1]

    cfg.d_patches = int(cfg.patch_size**2) * (cfg.latent_channel // 2 if cfg.split_latent else cfg.latent_channel)

    cfg.num_queries_list = [
        cfg.num_line_queries,
        cfg.num_vehicle_queries,
        # cfg.num_pedestrian_queries,
        # cfg.num_static_object_queries,
        # cfg.num_green_light_queries,
        # cfg.num_red_light_queries,
        #1,  # add ego query
    ]

    return cfg


def get_args():

    parser = argparse.ArgumentParser("Autoencoder for BEV")
    parser.add_argument('--project', type=str, default='autoencoder')
    parser.add_argument('--model_name', type=str, default='bev-autoencoder')
    parser.add_argument('--config', type=str, default='config.yaml')

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        cfg = yaml.safe_load(config_file)

    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg)
    args = set_args(cfg)
    
    print("Config:")
    print(args)

    for key, value in vars(args).items():
        setattr(args, key, value)
    
    args.gpus = torch.cuda.device_count()

    print_args(args)
    
    return args


# def set_remaining_args(args):
#     args.gpus = torch.cuda.device_count()
    

#     args.encoder_args = {
#         'encoder_type': args.backbone,
#         'resolution': args.resolution,
#         'use_lora': args.use_lora,
#         'lora_rank': args.lora_rank,
#         'finetune': args.finetune,
#     }

#     if args.aug:
#         args.rand_crop_and_resize = True
#         args.rand_flip = True
#         args.do_shuffle_cams = True


# def print_args(args):

#     print("====== Training ======")
#     print(f"project name: {args.project}\n")

#     print(f"backbone: {args.backbone}\n")

#     print(f"model: {args.model_name}\n")

#     print(f"resolution: {args.resolution}\n")

#     print(f"learning_rate: {args.learning_rate}")
#     print(f"batch_size: {args.batch_size}")
#     print(f"effective_batch_size: {args.batch_size * args.gradient_acc_steps}")
#     print(f"num_steps: {args.num_steps}")

#     print("====== ======= ======\n")

# def get_args():
#     parser = argparse.ArgumentParser("Robust Camera-Based BEV Segmentation by Adapting DINOV2")

#     parser.add_argument('--project', type=str, default='robust-bev')
#     parser.add_argument('--model_name', type=str, default='simplebev')

#     # === Model Related Parameters ===
#     parser.add_argument('--backbone', type=str, default="dinov2_s", choices=["res101", "dinov2_s", "dinov2_b", "dinov2_l"])

#     # finetuning
#     parser.add_argument('--finetune', action="store_true")

#     # adaptation setting
#     parser.add_argument('--use_lora', action="store_true")
#     parser.add_argument('--lora_rank', type=int, default=32)
#     parser.add_argument('--use_qkv', action="store_true")

#     parser.add_argument('--do_rgbcompress', action="store_true")

#     # === Data Related Parameters ===
#     parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset (e.g., /datasets/nuscenes)")
#     parser.add_argument('--version', type=str, default='trainval')
#     parser.add_argument('--res_scale', type=int, default=1)
#     parser.add_argument('--H', type=int, default=1600)
#     parser.add_argument('--W', type=int, default=900)
#     parser.add_argument('--resolution',  nargs='+', type=int, default=[224, 400])
#     parser.add_argument('--rand_crop_and_resize',  action="store_true")
#     parser.add_argument('--rand_flip',  action="store_true")
#     parser.add_argument('--cams',  nargs='+', type=str, default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])
#     parser.add_argument('--ncams', type=int, default=6)
#     parser.add_argument('--aug',  action="store_true")
#     parser.add_argument('--do_shuffle_cams',  action="store_true")
#     parser.add_argument('--refcam_id', type=int, default=1)
#     parser.add_argument('--bounds', nargs='+', type=int, default=[-50, 50, -5, 5, -50, 50])

#     # === Log Parameters ===
#     parser.add_argument('--log_freq', type=int, default=2000)
    
#     # === Training Related Parameters ===
#     parser.add_argument('--learning_rate', type=float, default=3e-4)
#     parser.add_argument('--weight_decay', type=float, default=1e-7)
#     parser.add_argument('--dropout', action="store_true")

#     parser.add_argument('--batch_size', type=int, default=2
#                         )
#     parser.add_argument('--gradient_acc_steps', type=int, default=1)
#     parser.add_argument('--num_steps', type=int, default=25000)
#     parser.add_argument('--num_epochs', type=int, default=10)
   
#     parser.add_argument('--seed', type=int, default=41)

#     # === Misc ===
#     parser.add_argument('--save_epoch', type=int, default=10)
#     parser.add_argument('--validate', action="store_true")
#     parser.add_argument('--evaluate_all_val', action="store_true")
#     parser.add_argument('--use_checkpoint', action="store_true")
#     parser.add_argument('--checkpoint_path', type=str, default=None)
#     parser.add_argument('--model_save_path', type=str, default='./checkpoints')

#     args = parser.parse_args()

#     set_remaining_args(args)

#     return args