import os
import time
import wandb
import datetime

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tqdm import tqdm


from nuscenes.prediction.helper import angle_diff
from nuscenes.prediction.input_representation.static_layers import color_by_yaw
import matplotlib.pyplot as plt
import numpy as np
import cv2


import utils
from read_args import get_args, print_args



def train_epoch(args, model, optimizer, scheduler, train_dataloader, val_dataloader, total_iter, val_info):
    
    total_loss = 0 

    loader = tqdm(train_dataloader, disable=(args.gpu != 0))

    for i, batch in enumerate(loader):
        model.train()

        logs = {} 

        # === Update time ====
        if (i) % args.gradient_acc_steps == 0:

            batch['features'] = batch['features'].cuda()
            for tgt_type in ['VEHICLES', 'LANES']:
                for k in ['vector', 'mask']:  
                    batch['targets'][tgt_type][k] = batch['targets'][tgt_type][k].clone().cuda() 

            batch['targets']['VEHICLES']['class'] = batch['targets']['VEHICLES']['class'].cuda()

            out = model(batch)

            loss = out['total_loss']

            loss /= args.gradient_acc_steps
            total_loss += loss.item()
            loss.backward()
    
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            total_iter += 1

            if args.gpu == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                mean_loss = total_loss / (i + 1)
                loader.set_description(f"lr: {lr:.6f} | loss: {mean_loss:.5f} | total_iter: {total_iter}")

                logs = {
                        'total iter': total_iter,
                        'loss': mean_loss, 
                        'lr':lr
                     }
                
                loss_details = out["loss_details"]
                for k, v in loss_details.items():
                    if  k != 'total_loss':
                        logs[f"{k}"] = v

                
                last_iter = (total_iter == (len(train_dataloader) * args.num_epochs // args.gradient_acc_steps) - 1)
                if total_iter % args.log_freq == 0 or last_iter:

                    # train_subset_loader = utils.get_random_subset_dataloader(train_dataloader.dataset, n=1000)
                    # train_logs, _ = eval(model, train_subset_loader)
                    # for k, v in train_logs.items():
                    #     logs[f"train/{k}"] = v

                    if args.evaluate_all_val:
                        val_logs, val_loss = eval(model, val_dataloader)
                        
                    else:
                        val_subset_loader = utils.get_random_subset_dataloader(val_dataloader.dataset, n=1000)
                        val_logs, val_loss = eval(model, val_subset_loader)

                    for k, v in val_logs.items():
                        logs[f"val/{k}"] = v
                    
                    logs["val/loss"] = val_loss
                    
                    # if val_logs['mIoU'] > val_info['best_val_miou']:
                    #     val_info['best_val_miou'] = val_logs['mIoU']
                    #     save_dict = {
                    #         "model": model.state_dict(),
                    #         "optimizer": optimizer.state_dict(),
                    #         "scheduler": scheduler.state_dict(),
                    #         "epoch": val_info["epoch"] + 1,
                    #         "args": args,
                    #     }
                    #     utils.save_on_master(save_dict, os.path.join(args.model_save_path, val_info['run_name'], f"best.pt"))
                    #     epoch = val_info['epoch']
                    #     print(f"Model saved at best epoch {epoch}")




        # === Just Calculate ====
        else:
            with model.no_sync():
                batch['features'] = batch['features'].cuda()
                for tgt_type in ['VEHICLES', 'LANES']:
                    for k in ['vector', 'mask']:  
                        batch['targets'][tgt_type][k] = batch['targets'][tgt_type][k].clone().cuda() 

                out = model(batch)

                loss = out['total_loss']

                loss /= args.gradient_acc_steps
                total_loss += loss.item()
                loss.backward()

             
        # === Log ===
        if args.gpu == 0:
            if logs:
                wandb.log(logs)


    statistics = {
        'loss': total_loss / (i + 1),
    }
   
    return statistics, total_iter, val_info


def uv_to_color(uv_mask):
    u = uv_mask[0]      # Z, X
    v = uv_mask[1]      # Z, X

    size = u.shape[-1]

    angle = (torch.atan2(v, u)) #% (2*np.pi)
    hue = ((((angle * 180) ) / (np.pi)) / 180 + 1) / 2 #  + np.pi / 2 + 0.25   #  + np.pi 2 * 

    color_mask = torch.zeros(3, size, size)

    for i in range(size):
        for j in range(size):
            if u[i,j] == 0 and v[i,j] == 0:
                color_mask[:, i, j] = torch.tensor([1.,1.,1.])
            else:
                color = plt.cm.hsv(hue[i, j].item())[:3] 
                color_mask[:, i, j] = torch.tensor(color)

    return color_mask


def visualize_vehicles(vehicle_vectors):

    points_list = []
    yaws = []
    img_h, img_w = 256, 256

    for idx in range(vehicle_vectors.shape[0]):
        vehicle_pred = vehicle_vectors[idx].detach().cpu()
        center_x, center_y = ((vehicle_pred[:2]) * 0.5 + 0.5)  * 256.0 + 0.5

        if center_x == 128.5 and center_y == 128.5:
            continue

        #width, length = (vehicle_pred[-2:] * 0.5 + 0.5)
        #yaw = vehicle_pred[2] * np.pi
        width, length = (vehicle_pred[[3,4]] * 0.5 + 0.5) * 0.5
        yaw = vehicle_pred[2] * np.pi

        yaws.append(yaw)

        if center_x < 0 or center_x >= 256. or center_y < 0 or center_y >= 256.:
            continue 
        
        width = width * 15 
        length = length * 15 
        pixel_width = (width) / 0.5
        pixel_length = (length) / 0.5
        half_width = pixel_width / 2
        half_length = pixel_length / 2

        top_left = (center_x - half_width, center_y - half_length)
        top_right = (center_x + half_width, center_y - half_length)
        bottom_left = (center_x - half_width, center_y + half_length)
        bottom_right = (center_x + half_width, center_y + half_length)

        points = np.array([top_left, top_right, bottom_right, bottom_left])

        angle_rad = yaw + np.pi / 2
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        rotated_points = np.dot(points - [center_x, center_y], rotation_matrix.T) + [center_x, center_y]
        rotated_points = rotated_points.astype(np.int32).reshape((-1, 2))
        
        points_list.append(rotated_points)


    vehicles_raster = torch.zeros(2, 256, 256).to(torch.float64)
    for i in range(len(points_list)):
        bbox = points_list[i] 
        bbox[:,1] = img_h - 1 - bbox[:,1] 
        poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [bbox], 1)

        theta = angle_diff(yaws[i],  0, 2*np.pi) 
        u = torch.cos(theta.clone()).item()     # between -1 and 1
        v = torch.sin(theta.clone()).item()     # between -1 and 1

        vehicles_raster[0][poly_mask == 1] = u  
        vehicles_raster[1][poly_mask == 1] = -v 

    color_mask = uv_to_color(vehicles_raster)  

    return (color_mask.permute(1, 2, 0) * 255.).numpy().astype('uint8')


def visualize_lanes(lane_vectors):

    image = np.full((256, 256, 3), 255.)

    lanes = (lane_vectors * 0.5 + 0.5) * 256.0 
    color_function=color_by_yaw

    for poses_along_lane in lanes:

        for start_pose, end_pose in zip(poses_along_lane[:-1], poses_along_lane[1:]):

            start_pixels = start_pose.detach().cpu() 
            end_pixels = end_pose.detach().cpu()     

            dx = end_pixels[0] - start_pixels[0]
            dy = end_pixels[1] - start_pixels[1]
            angle_radians = torch.atan2(dx, dy)

            start_pixels = start_pixels.numpy()
            end_pixels = end_pixels.numpy()
            
            start_pixels = (start_pixels[0].astype(int), start_pixels[1].astype(int))
            end_pixels = (end_pixels[0].astype(int), end_pixels[1].astype(int))

            if start_pixels == (128, 128) and end_pixels == (128, 128):
                continue

            angle = angle_diff(angle_radians.item(), 0,  2*np.pi) + np.pi
      
            color = color_function(0, angle)  
            cv2.line(image, start_pixels, end_pixels, color, thickness=2)

    return image.astype('uint8')



@torch.no_grad()
def eval(model, val_dataloader):

    model.eval()

    val_loader = tqdm(val_dataloader)
    
    total_loss = 0
    wandb_images = []

    indexes = [0, 260, 520, 895] #1500, 3200, 5000, 6000]

    for i, batch in enumerate(val_loader):

        if i == 2000:
            break

        batch['features'] = batch['features'].cuda()
        for tgt_type in ['VEHICLES', 'LANES']:
            for k in ['vector', 'mask']:  
                batch['targets'][tgt_type][k] = batch['targets'][tgt_type][k].clone().cuda() 

        batch['targets']['VEHICLES']['class'] = batch['targets']['VEHICLES']['class'].cuda()

        out = model(batch)

        total_loss += out['total_loss']
        #kl_metric = out['loss_details']['kl_metric']

        if len(wandb_images) < 5:
    
            if i not in indexes:
                continue

            vehicle_pred_vectors = out['vector']['VEHICLES']['vector'][0]
            vehicle_pred_classes = out['vector']['VEHICLES']['mask'][0]
            vehicle_gts = batch['targets']['VEHICLES']['vector'][0]  

            mask = torch.sigmoid(vehicle_pred_classes) > 0.3
            # probs = torch.softmax(vehicle_pred_classes, dim=-1)
            # pred_classes = torch.argmax(probs, dim=-1)
            # mask = (pred_classes != 0)
            vehicle_pred_vectors = vehicle_pred_vectors[mask]
            vehicles_pred_raster = visualize_vehicles(vehicle_pred_vectors)
            vehicles_gt_raster = visualize_vehicles(vehicle_gts)

            vehicles_overlap = 0.8 * vehicles_pred_raster + 0.2 * vehicles_gt_raster

            lane_pred_vectors = out['vector']['LANES']['vector'][0]
            lane_pred_classes = out['vector']['LANES']['mask'][0]
            lane_gts = batch['targets']['LANES']['vector'][0]
            
            mask = torch.sigmoid(lane_pred_classes) > 0.9
            lane_pred_vectors = lane_pred_vectors[mask]

            lanes_pred_raster = visualize_lanes(lane_pred_vectors)
            lanes_gt_raster = visualize_lanes(lane_gts)

            lanes_overlap = 0.8 * lanes_pred_raster + 0.2 * lanes_gt_raster

            images = [vehicles_gt_raster, vehicles_pred_raster, vehicles_overlap, lanes_gt_raster, lanes_pred_raster, lanes_overlap]

            top_row = np.concatenate(images[:3], axis=1)
            bottom_row = np.concatenate(images[3:], axis=1)
            concat = np.concatenate([top_row, bottom_row], axis=0) 
            
            wandb_images.append(wandb.Image(concat, caption=f"GT - Pred - Overlay"))

        # ===  Segmentation Evaluation ===
        metric_desc = f"Val: "

        # === Logger ===
        val_loader.set_description(metric_desc)
        # === === ===

    # === Evaluation Results ====
    total_loss =  total_loss / (i+1)
    logs = out['loss_details']
    logs['pred_boxes'] = wandb_images

    return logs, total_loss



def main_worker(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print_args(args)
   
    # === Model ===
    model = utils.init_model(args)

    utils.print_model_summary(args, model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Dataloaders ====
    trainset, valset = utils.get_datasets(args)
    train_dataloader, val_dataloader = utils.get_dataloaders(args, trainset, valset)

    # === Epochs ===
    args.num_epochs = args.num_epochs #((args.num_steps * args.gradient_acc_steps) // len(train_dataloader))
    args.num_steps = args.num_epochs * len(train_dataloader) // args.gradient_acc_steps

    print('#########################################')
    print('It will take {} epochs to complete'.format(args.num_epochs))
    print('#########################################')

    # === Training Items ===
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = utils.get_scheduler(args, optimizer)

    print('#########################################')
    print(f"Optimizer and schedulers ready.")
    print('#########################################')

    # === Misc ===
    run_name = utils.get_run_name(args)

    if args.gpu == 0:
        utils.init_logger(args, run_name)

    # === Load from Checkpoint ===
    to_restore = {"epoch": 0}
    if args.use_checkpoint:
        utils.restart_from_checkpoint(args, 
                                      run_variables=to_restore, 
                                      model=model, 
                                      optimizer=optimizer, 
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]

    start_time = time.time()

    dist.barrier()

    # ========================== Val =================================== #
    if args.validate:

        print("Starting evaluation!")
        if args.gpu == 0:
            val_logs, val_loss = eval(model, val_dataloader)
            logs = {}
            for k, v in val_logs.items():
                logs[f"val/{k}"] = v
                logs["val/loss"] = val_loss
            wandb.log(logs)

        dist.barrier()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
        dist.destroy_process_group()
        return

    # ========================== Train =================================== #

    if args.gpu == 0:
        if not os.path.exists(os.path.join(args.model_save_path, run_name)):
            os.makedirs(os.path.join(args.model_save_path, run_name))

    print("Starting training!")

    total_iter = 0

    val_info = {
            'best_val_miou': 0,
            'epoch': start_epoch,
            'run_name' : run_name
        }

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        

        statistics, total_iter, val_info = train_epoch(args, 
                                                            model, 
                                                            optimizer, 
                                                            scheduler, 
                                                            train_dataloader, 
                                                            val_dataloader, 
                                                            total_iter,
                                                            val_info
                                                            ) 
                                                            

        if args.gpu == 0:

            if epoch % args.save_epoch == 0 or epoch == args.num_epochs - 1:

                # === Save Checkpoint ===
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "args": args,
                }

                utils.save_on_master(save_dict, os.path.join(args.model_save_path, run_name, f"{epoch}.pt"))
                print(f"Model saved at epoch {epoch}")

        dist.barrier()

        print("===== ===== ===== ===== =====")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    main_worker(args)