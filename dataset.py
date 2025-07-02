"""
    This code is adapted from https://github.com/aharley/simple_bev/nuscenesdataset.py
"""


import numpy
import numpy as np
import random
import os
from PIL import Image
import cv2
import io
import torch
import torchvision
from torch.utils.data import Dataset
import matplotlib.path as mplPath
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.map_expansion.map_api import NuScenesMap
from functools import reduce
from nuscenes.utils.geometry_utils import transform_matrix

import torch.nn.functional as F
import torchvision.transforms.functional as TF

#from nuscenesdataset import get_nusc_maps, fetch_nusc_map2, add_ego2

from bev_utils import geom, py, vox, improc
import matplotlib.pyplot as plt

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer, draw_lanes_in_agent_frame, get_lanes_in_radius, draw_lanes_on_image, color_by_yaw, get_patchbox
from nuscenes.prediction.input_representation.utils import get_crops, get_rotation_matrix, convert_to_pixel_coords
from nuscenes.prediction.helper import angle_of_rotation, angle_diff


discard_invisible = False

totorch_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))

COUNT = 0

MAP_NAME_ID_ABBR = [
    (0, "us-nv-las-vegas-strip", "LAV"),
    (1, "us-pa-pittsburgh-hazelwood", "PGH"),
    (2, "sg-one-north", "SGP"),
    (3, "us-ma-boston", "BOS"),
]

MAP_NAME_TO_ID = {name: id for id, name, abbr in MAP_NAME_ID_ABBR}
MAP_ID_TO_NAME = {id: name for id, name, abbr in MAP_NAME_ID_ABBR}
MAP_ID_TO_ABBR = {id: abbr for id, name, abbr in MAP_NAME_ID_ABBR}

_LINE_X = 0
_LINE_Y = 1
_VEHICLE_X = 2
_VEHICLE_Y = 3
_PEDESTRIAN_X = 4
_PEDESTRIAN_Y = 5
_STATIC_OBJECT_X = 6
_STATIC_OBJECT_Y = 7
_GREEN_LIGHT_X = 8
_GREEN_LIGHT_Y = 9
_RED_LIGHT_X = 10
_RED_LIGHT_Y = 11


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img

def move_refcam(data, refcam_id):

    data_ref = data[refcam_id].clone()
    data_0 = data[0].clone()

    data[0] = data_ref
    data[refcam_id] = data_0

    return data


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def rotate_points_2d(points, rotation_mat):
    num_points = points.shape[0]
    ones = np.ones((num_points, 1))
    points_hom = np.hstack([points, ones])  # shape (N, 3)
    
    rotated_hom = points_hom.dot(rotation_mat.T)  # (N,3) * (3,2) = (N,2)
    
    return rotated_hom


def resample_lane_pts(lane_points, num_samples=20):
    """
    Resample a lane (list/array of (x, y) points) into 'num_samples' equally spaced points.
    
    :param lane_points: Nx2 array-like of x, y coordinates
    :param num_samples: number of desired sample points in output
    :return: (num_samples x 2) array of resampled x, y coordinates
    """
    lane_points = np.asarray(lane_points, dtype=float)
        
    diffs = lane_points[1:] - lane_points[:-1]      # (N-1, 2)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))   # (N-1,)
    
    cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)  # shape: (N,)
    total_length = cum_lengths[-1]
    
    target_distances = np.linspace(0, total_length, num=num_samples, endpoint=True)
    
    resampled_points = []
    
    seg_index = 0
    for dist in target_distances:
        while seg_index < len(cum_lengths) - 2 and dist > cum_lengths[seg_index + 1]:
            seg_index += 1
        
        seg_start_length = cum_lengths[seg_index]
        seg_end_length   = cum_lengths[seg_index + 1]
        
        seg_frac = (dist - seg_start_length) / (seg_end_length - seg_start_length + 1e-9)
        seg_frac = np.clip(seg_frac, 0.0, 1.0)  
        
        pt_start = lane_points[seg_index]
        pt_end   = lane_points[seg_index + 1]
        
        interp_xy = pt_start + seg_frac * (pt_end - pt_start)
        resampled_points.append(interp_xy)
    
    return np.array(resampled_points)

class NuscenesDataset(Dataset):

    def __init__(
        self,
        args,
        nusc
    ):
        self.is_train = args.is_train

        self.rand_crop_and_resize = args.rand_crop_and_resize
        self.final_dim = args.bev_resolution
        #self.img_resolution = args.img_resolution
        if self.rand_crop_and_resize:
            self.resize_lim = [0.8,1.2]
            self.crop_offset = int(self.final_dim[0]*(1-self.resize_lim[0]))
        else:
            self.resize_lim  = [1.0,1.0]
            self.crop_offset = 0
        self.H = args.H
        self.W = args.W
        self.cams = args.cams
        self.ncams = args.ncams
        self.do_shuffle_cams = args.do_shuffle_cams
        self.refcam_id = args.refcam_id
        self.bounds = args.bounds

        self.version = args.version

        self.dataroot = args.dataset_path

        self.X, self.Y, self.Z = (self.final_dim[0], 8, self.final_dim[1]) #args.voxel_size
        self.nusc = nusc

        split = 'train' if args.is_train else 'val'
        scenes = create_splits_scenes()[split]
            
        samples = [samp for samp in self.nusc.sample]
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        self.samples = samples
        self.city_names = self.get_city_names(samples)

        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = args.bounds

        grid_conf = { # note the downstream util uses a different XYZ ordering
            'xbound': [XMIN, XMAX, (XMAX-XMIN)/float(self.X)],
            'ybound': [ZMIN, ZMAX, (ZMAX-ZMIN)/float(self.Z)],
            'zbound': [YMIN, YMAX, (YMAX-YMIN)/float(self.Y)],
        }
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.nsweeps = 1
        self.use_radar_filters = False

        # the scene centroid is defined wrt a reference camera,
        # which is usually random
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 # down 1 meter
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float().cuda()

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=scene_centroid,
            bounds=self.bounds,
            assert_cube=False)
        
        #self.vehicle_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

        self.vehicle_classes = ['vehicle.bicycle', 
                                'vehicle.bus', 
                                'vehicle.car', 
                                'vehicle.construction', 
                                'vehicle.emergency',
                                'vehicle.motorcycle', 
                                'vehicle.trailer', 
                                'vehicle.truck']

        self.helper = PredictHelper(nusc)
        self.lane_rasterizer = StaticLayerRasterizer(self.helper)
        
        self.lane_rasterizer.meters_ahead = 64.0
        self.lane_rasterizer.meters_behind = 64.0
        self.lane_rasterizer.meters_left = 64.0
        self.lane_rasterizer.meters_right = 64.0
        self.lane_rasterizer.resolution = 0.5
        
        self.image_side_length = 2 * max(self.lane_rasterizer.meters_ahead, self.lane_rasterizer.meters_behind,
                                    self.lane_rasterizer.meters_left, self.lane_rasterizer.meters_right)
        self.image_side_length_pixels = int(self.image_side_length / self.lane_rasterizer.resolution)


        self.max_num_vehicles = args.num_vehicles
        self.max_num_lines = args.num_lines
        self.vehicle_drop_rate = args.vehicle_drop_rate


        self.multiclass = args.multiclass
        

    
    def get_city_names(self, samples):

        city_names = []

        for i , sample in enumerate(samples):
            scene = self.nusc.get('scene', sample['scene_token'])
            city_name = self.nusc.get('log', scene['log_token'])['location']
            if 'singapore' in city_name:
                city_name = 'singapore'
            else:
                city_name = 'boston'

            city_names.append(city_name)

        return city_names


        
    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):
   
        sample = self.samples[idx]
        city_name = self.city_names[idx] 
        city_token = 0 if city_name == 'boston' else 1

        # ============= Extract Vehicle Masks ===================

        lrtlist_, boxlist_, vislist_, tidlist_, ptslist_, yawlist_, sizelist_, bbox2dlist_,  bbox3dlist_, velocities_, classes_, bboxes_md = self.get_lrtlist(sample)
        
        points = ptslist_  
        heading_angles = yawlist_
        # speeds = velocities_

        vehicles_raster = torch.zeros(2, self.final_dim[0], self.final_dim[1]).to(torch.float64)
        for i in range(len(points)):
            bbox = points[i].cpu().numpy().astype(np.int32)  
            bbox[:,1] = self.final_dim[0] - 1 - bbox[:,1] 
            poly_mask = np.zeros((self.final_dim[0], self.final_dim[1]), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [bbox], 1)

            # speed = speeds[i]
            theta =  angle_diff(heading_angles[i] * np.pi ,  0, 2*np.pi) #(heading_angles[i] * np.pi + np.pi) #/ 2)   # 0, 2*pi
            u = torch.cos(theta.clone()).item()    # between -1 and 1
            v = torch.sin(theta.clone()).item()    # between -1 and 1
        
            vehicles_raster[0][poly_mask == 1] = u  
            vehicles_raster[1][poly_mask == 1] = -v 
        
        color_mask = self.uv_to_color(vehicles_raster)                               # 3, Z, X
        vehicles_raster = vehicles_raster.permute(1, 2, 0)                           # Z, X, 2

        # uv_mask --> between 0 and 1     (or -1 to 1)
        # vehicles_raster = (vehicles_raster + 1) / 2             # 2, 256, 256

        vehicles_vector = torch.zeros(self.max_num_vehicles, bboxes_md.shape[1])   
        vehicles_labels = torch.zeros(self.max_num_vehicles)
        vehicles_classes = torch.zeros(self.max_num_vehicles) 
        last_idx = min(len(bboxes_md), self.max_num_vehicles)

        vehicles_vector[:last_idx] = (bboxes_md[:last_idx])
        vehicles_labels[:last_idx] = True
        vehicles_classes[:last_idx] = classes_[:last_idx].view(-1)

        # ================= Extract Lane Masks =============

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = egopose['translation']
        rot = egopose['rotation']
        sample_token = sample['token']

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        sensor_translation = np.array(calibrated_sensor['translation'])
        sensor_rotation = Quaternion(calibrated_sensor['rotation'])
        
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        
        sensor_global_rotation = ego_rotation * sensor_rotation 
        sensor_global_translation = ego_rotation.rotate(sensor_translation) + ego_translation

        map_name = self.lane_rasterizer.helper.get_map_name_from_sample_token(sample_token)
        
        x, y = sensor_global_translation[:2]                                                                          # ego trans
        yaw = quaternion_yaw(sensor_global_rotation) + np.pi/2
        patchbox = get_patchbox(x, y, self.image_side_length)

        images = []
        
        agent_x, agent_y = x, y  
        agent_yaw = yaw 
        radius=64
        image_resolution=self.lane_rasterizer.resolution
        discretization_resolution_meters=1
        map_api=self.lane_rasterizer.maps[map_name]
        
        agent_pixels = int(self.image_side_length_pixels / 2), int(self.image_side_length_pixels / 2)
        base_image = np.zeros((self.image_side_length_pixels, self.image_side_length_pixels, 3))
        
        lanes = get_lanes_in_radius(agent_x, agent_y, radius, discretization_resolution_meters, map_api)

        image = base_image
        agent_global_coords = (agent_x, agent_y)
        agent_yaw_in_radians = agent_yaw
        agent_pixels = agent_pixels
        resolution = image_resolution
        color_function= color_by_yaw
        rotation_mat = get_rotation_matrix(base_image.shape, agent_yaw)
        lanes_eliminated = []
        mask = np.zeros((self.image_side_length_pixels, self.image_side_length_pixels, 1))

        lane_tokens = map_api.get_records_in_radius(agent_x, agent_y, radius, ['lane'])['lane']
        # divider_mask = np.full((256, 256, 3), 255.) # np.full((256, 256, 3), 255.)

        lane_dividers_raster = np.zeros((self.image_side_length_pixels, self.image_side_length_pixels, 2))
        lane_dividers_in_frame = []
        lane_dividers_eliminated = []
        for lane_token in lane_tokens:
            
            lane_metadata = [record for record in map_api.lane if record['token'] == lane_token][0]

            cur_poses_along_lane = map_api.get_arcline_path(lane_token)

            for side in ['left_lane_divider_segment_nodes', 'right_lane_divider_segment_nodes']:
                lane_dividers_eliminated.append([])
                segment_nodes = lane_metadata.get(side, [])
                if len(segment_nodes) < 2:
                    continue

                lane_start_poses = np.array([pose['start_pose'] for pose in cur_poses_along_lane] + [pose['end_pose'] for pose in cur_poses_along_lane])
                lane_start_xy = lane_start_poses[:, :2] 
                lane_start_yaws = lane_start_poses[:, 2] 

                divider_coords = [(node['x'], node['y']) for node in segment_nodes]
        
                for i in range(0, len(divider_coords)-1):
                    start = divider_coords[i]
                    end = divider_coords[i + 1]

                    dists = np.linalg.norm(lane_start_xy - start, axis=1)
                    closest_idx = np.argmin(dists)
                    closest_yaw = lane_start_yaws[closest_idx]
                    yaw = closest_yaw

                    #dx = end[0] - start[0]
                    #dy = (end[1]) - (start[1])
                    #yaw = np.arctan2(dx, dy)
        
                    start_pixel = convert_to_pixel_coords(start, agent_global_coords, agent_pixels, resolution)
                    end_pixel   = convert_to_pixel_coords(end, agent_global_coords, agent_pixels, resolution)
        
                    start_pixels = (start_pixel[1], start_pixel[0])
                    end_pixels   = (end_pixel[1], end_pixel[0])
                    
                    start_pixels = rotate_points_2d(np.array([start_pixels]), rotation_mat)[0].astype(int)
                    end_pixels   = rotate_points_2d(np.array([end_pixels]), rotation_mat)[0].astype(int)

                    angle = angle_diff(agent_yaw_in_radians,  yaw, 2*np.pi) + np.pi # [-pi, +pi] + pi --> [0, 2*pi]
                    u = np.cos(angle)    # -1, 1
                    v = np.sin(angle)    # -1, 1

                    if start_pixels[0] >= self.image_side_length_pixels or start_pixels[0] < 0 or start_pixels[1] >= self.image_side_length_pixels or start_pixels[1] < 0:
                        continue

                    lane_dividers_raster[start_pixels[1], start_pixels[0]] = np.array([u, v])   
                    # lane_dividers_raster[end_pixel[1], end_pixel[0]] = np.array([u, v])

                    start_pixels = tuple(start_pixels.astype(int))
                    lane_dividers_eliminated[-1].append(start_pixels)

                    # color = color_function(agent_yaw_in_radians, yaw)  
                    # cv2.line(divider_mask, start_pixels, end_pixels, color, thickness=2)

                if end_pixels[0] < self.image_side_length_pixels and end_pixels[0] >= 0 and end_pixels[1] < self.image_side_length_pixels and end_pixels[1] >= 0:
                    lane_dividers_eliminated[-1].append(end_pixels)

                cur_lane_div = lane_dividers_eliminated[-1]
                if len(cur_lane_div) < 2:
                    for pixel in cur_lane_div: 
                        lane_dividers_raster[pixel[1], pixel[0]][0] = 0 
                        lane_dividers_raster[pixel[1], pixel[0]][1] = 0 
                    continue
                lane_dividers_in_frame.append(cur_lane_div)
        
        lane_dividers_raster = torch.from_numpy(lane_dividers_raster)  

        lane_dividers_vector = torch.zeros(self.max_num_lines, 10, 2)  
        lane_divider_labels = torch.zeros(self.max_num_lines)
        
        for lane_idx, lane in enumerate(lane_dividers_in_frame):
            if lane_idx >= self.max_num_lines:
                break
            lane_pts_sampled = resample_lane_pts(lane, num_samples=10)
            lane_dividers_vector[lane_idx] = (torch.from_numpy(lane_pts_sampled) / self.final_dim[0]) * 2 - 1

        last_idx = min(len(lane_dividers_in_frame), self.max_num_lines)
        lane_divider_labels[: last_idx] = True

        
        num_line_poses = 20
        lanes_raster = np.zeros((self.image_side_length_pixels, self.image_side_length_pixels, 2))
        lanes_in_frame = []

        for poses_along_lane in lanes.values():

            lanes_eliminated.append([])
            
            for start_pose, end_pose in zip(poses_along_lane[:-1], poses_along_lane[1:]):
        
                start_pixels = convert_to_pixel_coords(start_pose[:2], agent_global_coords, agent_pixels, resolution)
                end_pixels = convert_to_pixel_coords(end_pose[:2], agent_global_coords, agent_pixels, resolution)
        
                start_pixels = (start_pixels[1], start_pixels[0])
                end_pixels = (end_pixels[1], end_pixels[0])
        
                rotation_mat = get_rotation_matrix(base_image.shape, agent_yaw)
                
                start_array = np.array(start_pixels, dtype=np.float32).reshape(1, 2)
                end_array   = np.array(end_pixels,   dtype=np.float32).reshape(1, 2)
                
                rotated_start = rotate_points_2d(start_array, rotation_mat)[0]
                rotated_end   = rotate_points_2d(end_array,   rotation_mat)[0]
                
                rotated_start = tuple(rotated_start.astype(int))
                rotated_end   = tuple(rotated_end.astype(int))
        
                if rotated_start[0] >= self.image_side_length_pixels or rotated_start[0] < 0 or rotated_start[1] >= self.image_side_length_pixels or rotated_start[1] < 0:
                    continue
                
                angle = angle_diff(agent_yaw_in_radians,  start_pose[2], 2*np.pi) + np.pi # [-pi, +pi] + pi --> [0, 2*pi]
                u = np.cos(angle)    # -1, 1
                v = np.sin(angle)    # -1, 1
                lanes_raster[rotated_start[1], rotated_start[0]] = np.array([u, v])       # encoding relative orientation
        
                lanes_eliminated[-1].append(rotated_start)
                
                #color = color_function(agent_yaw_in_radians, start_pose[2])
        
                #cv2.line(image, rotated_start, rotated_end, color, thickness=2)
                
            cur_lane = lanes_eliminated[-1]
            if len(cur_lane) < 3:
                for pixel in cur_lane: 
                    lanes_raster[pixel[1], pixel[0]][0] = 0 
                    lanes_raster[pixel[1], pixel[0]][1] = 0 
                continue
            lanes_in_frame.append(cur_lane)

        lanes_raster = torch.from_numpy(lanes_raster)   # 256, 256, 4

        lanes_vector = torch.zeros(self.max_num_lines, num_line_poses, 2)  # len(lanes_in_frame)
        lanes_labels = torch.zeros(self.max_num_lines)
        for lane_idx, lane in enumerate(lanes_in_frame):
            if lane_idx >= self.max_num_lines:
                break
            lane_pts_sampled = resample_lane_pts(lane, num_samples=num_line_poses)
            lanes_vector[lane_idx] = (torch.from_numpy(lane_pts_sampled) / self.final_dim[0]) * 2 - 1

        last_idx = min(len(lanes_in_frame), self.max_num_lines)
        lanes_labels[: last_idx] = True

        raster = torch.zeros(256, 256, 6)
        raster[..., :2] = lanes_raster
        raster[..., 2:4] = vehicles_raster
        raster[..., 4:] = lane_dividers_raster

        target = {
            'features': raster.permute(2,0,1),                  # Z, X, 4
            'city_token': city_token,            # 1
            'color_mask' : color_mask,
            'targets':{
                        'LANES': {
                                    'vector': lanes_vector,    # L, 20 ,2 
                                    'mask' : lanes_labels
                        },
                        'LANE_DIVIDERS' :
                        {
                            'vector': lane_dividers_vector,    # L, 10 ,2 
                            'mask' : lane_divider_labels
                            
                        },
                        'VEHICLES': {
                                    'vector': vehicles_vector,  # V, 5
                                    'mask' : vehicles_labels,
                                    'class': vehicles_classes

                        }
            }
        }

        return  target
    

    def uv_to_color(self, uv_mask):
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
    

    def choose_cams(self):
        if self.is_train and self.ncams < len(self.cams):
            cams = np.random.choice(self.cams, 
                                    self.ncams,
                                    replace=False)
        else:
            cams = self.cams
        return cams


    def choose_ref_cam(self):

        if self.is_train and self.do_shuffle_cams:
            # randomly sample the ref cam
            refcam_id = np.random.randint(1, self.ncams)# len(self.cams))
        else:
            refcam_id = self.refcam_id

        return refcam_id

    def sample_augmentation(self):
        fH, fW =  self.img_resolution
        resize_dims = (fW, fH)
        crop_h = 0
        crop_w = 0
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop


    def get_image_data(self, sample, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            samp = self.nusc.get('sample_data', sample['data'][cam])

            imgname = os.path.join(self.dataroot, samp['filename'])
            img = Image.open(imgname)
            W, H = img.size

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])
            
            # images are not used
            resize_dims, crop = self.sample_augmentation()

            sx = resize_dims[0]/float(W)
            sy = resize_dims[1]/float(H)

            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = img_transform(img, resize_dims, crop)
            imgs.append(totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

            
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),torch.stack(intrins))


    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        # to ego-frame
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        #print('EGO ', trans, 'rot' , egopose['rotation'])
        # in degrees
        ego_yaw_trans = Quaternion(egopose['rotation']).inverse.yaw_pitch_roll[0] * 180 / np.pi  

        lidar_token = rec["data"]["LIDAR_TOP"]
        lidar_path, boxes, _ = self.nusc.get_sample_data(lidar_token)

        boxes_md = []
        classes = []
        ptslistmd = []
        yawlistmd = []
        for box in boxes:
            if "vehicle" in box.name:
                tok = box.token
                inst = self.nusc.get('sample_annotation', tok)
                if int(inst['visibility_token']) in [1, 2, 3]:  # invisible
                    continue

                box_center = box.center

                if box_center[0] >= 64 or box_center[0] <= -64 or box_center[1] >= 64 or box_center[1] <= -64:
                    continue

                pts = box.bottom_corners()[:2].T    # Bottom corners. First two face forward, last two face backwards.
                pts = np.round((pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
                    
                ptslistmd.append(torch.from_numpy(pts))

                pixel_center = np.array([(box_center[1]), (box_center[0])])  # 0, 256 

                pixel_center = np.array([(pixel_center[1] * 2 + 128.), (pixel_center[0] * 2 + 128.)])  # 0, 256
                pixel_center = (pixel_center / self.final_dim[0]) * 2 - 1

                pixel_center_z = np.clip(box_center[2], -5.0, 5.0) / 5.0
                # assert -1 <= pixel_center[0] <= 1 and -1 <= pixel_center[1] <= 1, f'Invalid pixel center: {pixel_center}'

                size = box.wlh
                pixel_size = (np.clip(size[:3], 0, 15.0) / 15.0) * 2 - 1    # -1, 1
                # assert -1 <= pixel_size[0] <= 1 and -1 <= pixel_size[1] <= 1, f'Invalid pixel size: {pixel_size}'
            
                rot2 = box.orientation.yaw_pitch_roll[0] - np.pi / 2        # this is in radians
                # find the constant multiplyer of pi
                # rot2 = (rot2 + np.pi) / (2 * np.pi)   # -1, 1
                k = rot2 * 180 / np.pi  
                if k < -180.0:
                    k += 360
                elif k > 180.0:
                    k -= 360
                k = k / 180.0 
                assert -1 <= k <= 1, f'Invalid yaw: {k}'
                yawlistmd.append(torch.tensor(k))
                box_md = np.array([pixel_center[0], pixel_center[1], k, pixel_size[1], pixel_size[0], pixel_center_z, pixel_size[2]])
                box_md = torch.from_numpy(box_md).float()
                boxes_md.append(box_md)

                if self.multiclass:
                    vehicle_class = '.'.join(inst['category_name'].split('.')[:2])
                    class_idx = self.vehicle_classes.index(vehicle_class) + 1
                else:
                    class_idx = 0
                # self.vehicle_classes.index(vehicle_class)
                classes.append(torch.Tensor([class_idx])) 
                # print(box.name, box.center, box.wlh, -box.orientation.yaw_pitch_roll[0]- np.pi / 2)
        boxes_md = torch.stack(boxes_md, dim=0) if len(boxes_md) > 0 else torch.zeros((0, 7))
        ptslist =  torch.stack(ptslistmd, dim=0) if len(ptslistmd) > 0 else torch.zeros((0, 4, 2))
        yawlist =  torch.stack(yawlistmd, dim=0) if len(yawlistmd) > 0 else torch.zeros((0))
        classes = torch.stack(classes, dim=0) if len(classes) > 0 else torch.zeros((0))
        
        
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        # ptslist = []
        # yawlist = []
        sizelist = []
        bbox2dlist = []
        bbox3dlist = []
        velocities = []
        # classes = []
        # for tok in rec['anns']:
            
        #     inst = self.nusc.get('sample_annotation', tok)
        #     # NuScenes filter
        #     if ('vehicle' not in inst['category_name']): # or ('cycle' in inst['category_name']):
        #         continue
            
        #     if int(inst['visibility_token']) == 1:
        #         continue

        #     # randomly drop some instances during training
        #     if self.is_train and random.random() < self.vehicle_drop_rate:
        #         continue

        #     box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
        #     box.translate(trans)
        #     box.rotate(rot)
            
        #     pts = box.bottom_corners()[:2].T    # Bottom corners. First two face forward, last two face backwards.
        #     pts = np.round((pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
            
        #     pts[:, [1, 0]] = pts[:, [0, 1]]          # 4, 2 - switch x, y
        #     pts[:, 0] = self.X - 1 - pts[:, 0]       # horizontal flip
            
        #     if np.any(pts < 0) or np.any(pts > self.X):
        #         continue
            
        #     # NOTE: invisibility not important for DETR
        #     vislist.append(torch.tensor(1.0)) # visible
            
        #     center_pt = np.mean(pts,axis=0)  # 2
            
        #     ptslist.append(torch.from_numpy(pts))          

        #     r = box.rotation_matrix
        #     t = box.center
        #     l = box.wlh
        #     l = np.stack([l[1],l[0],l[2]])
        #     lrt = py.merge_lrt(l, py.merge_rt(r,t))
        #     lrt = torch.Tensor(lrt)
        #     lrtlist.append(lrt)
        #     ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
        #     rs = np.stack([ry*0, ry, ry*0])
        #     box_ = torch.from_numpy(np.stack([t,l,rs])).reshape(9)
        #     boxlist.append(box_)
            
        #     yaw = (180 / np.pi * box_[7] + ego_yaw_trans) # + rotate_angle)   # yaw in degrees
        #     if yaw < -180.0:
        #         yaw += 360
        #     elif yaw > 180.0:
        #         yaw -= 360
        #     yaw = yaw / 180.0  # normalized [-1, +1]. -1 --> front-right-back, +1 --> front-left-back
        #     assert -1 <= yaw <= 1, f'Invalid yaw: {yaw}'

        #     yawlist.append(yaw)
            
        #     size = torch.clamp(box_[3:6], min = 0, max = 15.0) / 15.0  # l, w, h

        #     velocity = self.nusc.box_velocity(tok)   # (vx ?, vy ?, vz)
        #     velocities.append(torch.Tensor([np.sqrt(velocity[0]**2 + velocity[1]**2)]))

        #     vehicle_class = '.'.join(inst['category_name'].split('.')[:2])
        #     class_idx = self.vehicle_classes.index(vehicle_class)
            
        #     # ['car', 'truck', 'construction_vehicle', 'bus', 
        #     # 'trailer', 'barrier', 'motorcycle', 'bicycle', 
        #     # 'pedestrian', 'traffic_cone']

        #     #class_idx = 0
        #     classes.append(torch.Tensor([class_idx + 1]))
        #     # NOTE: for now class_idx is 0.
            
        #     bbox2d = torch.tensor([center_pt[0],center_pt[1],yaw,size[0],size[1]], dtype = torch.float32)
        #     bbox2dlist.append(bbox2d)

        #     bbox3d = torch.zeros(7)
        #     bbox3d[:-1] = box_[:6]
        #     bbox3d[-1] = yaw
        #     bbox3d[[0,1]] = bbox3d[[1,0]] 
        #     bbox3d[0] = (-1) * bbox3d[0]    
        #     bbox3dlist.append(bbox3d)

            
        # if len(ptslist):
        #     vislist = torch.stack(vislist, dim=0)
        #     ptslist = torch.stack(ptslist, dim=0)
        #     yawlist = torch.stack(yawlist, dim=0)
        #     velocities = torch.stack(velocities, dim=0)
        #     classes = torch.stack(classes, dim=0)
        #     # sizelist = torch.stack(sizelist, dim=0)
        #     bbox2dlist = torch.stack(bbox2dlist, dim=0)
        #     bbox3dlist = torch.stack(bbox3dlist, dim=0)
        #     # tidlist = torch.stack(tidlist, dim=0)
        # else:
        #     lrtlist = torch.zeros((0, 19))
        #     boxlist = torch.zeros((0, 9))
        #     vislist = torch.zeros((0))
        #     ptslist = torch.zeros((0, 4, 2))
        #     yawlist = torch.zeros((0))
        #     sizelist = torch.zeros((0, 2))
        #     bbox2dlist = torch.zeros((0, 5))
        #     bbox3dlist = torch.zeros((0, 7))
        #     velocities =  torch.zeros((0))
        #     classes =  torch.zeros((0))
        #     # tidlist = torch.zeros((0))

        return lrtlist, boxlist, vislist, tidlist, ptslist, yawlist, sizelist, bbox2dlist, bbox3dlist, velocities, classes, boxes_md
           
    
    def get_seg_bev(self, pts_list):
        
        N = pts_list.shape[0]
        vehicle_masks = np.zeros((N, self.X,  self.Z))
        
        for n in range(N):
            pts = pts_list[n].detach().numpy() # 4, 2
            cv2.fillPoly(vehicle_masks[n], [pts.astype(np.int32)], n + 1.0)


        seg = torch.Tensor(vehicle_masks)   # N, Z, X 

        return seg





class NuScenesDatasetWrapper:
    '''
        Initialize training and validation datasets.
    '''
    def __init__(self, args, nusc=None):
        self.args = args

        print('Loading NuScenes version', args.version, 'from', args.dataset_path)

        if nusc is None:
            print('Load it from scratch')
            self.nusc = NuScenes(
                        version='v1.0-{}'.format(args.version),
                        dataroot=args.dataset_path, 
                        verbose=True
                        )
        else:
           self.nusc = nusc
        print('Done loading NuScenes version', args.version)

    
    def train(self):
        self.args.is_train = True
        return NuscenesDataset(self.args, self.nusc)
    
    def val(self):
        self.args.is_train = False
        return NuscenesDataset(self.args, self.nusc)
    

class Args:
    def __init__(self):
        # Basic settings
        self.is_train = True
        self.rand_crop_and_resize = True

        # Resolution / dimensions
        self.bev_resolution = (256, 256)    # final BEV resolution (X, Z)
        self.img_resolution = (900, 1600)   # input image resolution
        self.H = 900
        self.W = 1600

        # Cameras and bounds
        self.cams = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        self.ncams = 6
        self.do_shuffle_cams = False
        self.refcam_id = 0
        self.bounds = [-64, 64, -5, 5, -64, 64]

        # Dataset information
        self.version = "trainval"
        self.dataset_path = "/datasets/nuscenes"

        # Derived attributes or placeholders (optional)
        self.X, self.Y, self.Z = (self.bev_resolution[0], 8, self.bev_resolution[1])
        self.split = "train" if self.is_train else "val"

        self.config = './configs/base.yaml'

args = Args()