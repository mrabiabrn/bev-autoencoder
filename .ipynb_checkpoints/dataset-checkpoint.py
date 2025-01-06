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


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

def fetch_nusc_map2(rec, nusc_maps, nusc, scene2map, car_from_current):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    global_from_car = transform_matrix(egopose['translation'],
                                       Quaternion(egopose['rotation']), inverse=False)

    trans_matrix = reduce(np.dot, [global_from_car, car_from_current])

    rot = np.arctan2(trans_matrix[1,0], trans_matrix[0,0])
    center = np.array([trans_matrix[0,3], trans_matrix[1,3], np.cos(rot), np.sin(rot)])

    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    poly_names = ['drivable_area', 'road_segment', 'lane']
    poly_names = ['lane','road_segment']           
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
        50.0, poly_names, line_names)

    return poly_names, line_names, lmap



def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )
            
    global_polys = polys.copy()

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys, global_polys

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




class NuscenesDataset(Dataset):

    def __init__(
        self,
        args,
        nusc
    ):
        self.is_train = args.is_train

        self.rand_crop_and_resize = args.rand_crop_and_resize
        self.final_dim = args.bev_resolution
        self.img_resolution = args.img_resolution
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
        
        self.nusc_maps = get_nusc_maps(args.dataset_path)
        self.scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']
            
        self.conditional = args.conditional
        self.gt_folder_name = args.gt_folder_name
        self.guidance_scale = args.guidance_scale
        self.vehicle_drop_rate = args.vehicle_drop_rate

    
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

        cache_path = '/home/mbarin22/projects/bev-layout-diffusion/bev-layout-diffusion/bevgt/' + self.gt_folder_name + '/' + str(idx) + '.npz'  # /bev-layout-diffusion/bevgt/ /bev-layout-diffusion-old/

        sample = self.samples[idx]
        city_name = self.city_names[idx]    
        direction = 'left' if city_name == 'singapore' else 'right'


        poly_color = (1.00, 200/255., 0.)
        road_divider_color = (0.0, 0.0, 1.0)  # RGB for road divider
        lane_divider_color = (1.0, 0.0, 1.0)  # RGB for lane divider

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            map_vis = data['map_rgb']
            vehicle_masks = data['vehicle_masks']
            
            map_vis = torch.from_numpy(map_vis)                 # 3, 400, 400
            _, mH, mW = map_vis.shape
            vehicle_masks = torch.from_numpy(vehicle_masks)     # 1, Z, X

        else:
            
            print('Generating cache for idx:', idx)
    
            lrtlist_, boxlist_, vislist_, tidlist_, ptslist_, yawlist_, sizelist_, bbox2dlist_ = self.get_lrtlist(sample)
            
            # ================= Extract MAP =================
            rec = self.samples[idx]
            egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
            map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]
    
            rot = Quaternion(egopose['rotation']).rotation_matrix
            rot = numpy.arctan2(rot[1, 0], rot[0, 0])
            center = numpy.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])
    
            poly_names = ['road_segment', 'lane']
            line_names = ['road_divider', 'lane_divider']
            lmap, global_polys = get_local_map(self.nusc_maps[map_name], center,
                                50.0, poly_names, line_names)
    
            bx, dx = self.bx[:2], self.dx[:2]
            fig = plt.figure(figsize=(4,4))
            ax = fig.gca()
            ax.axis('off')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            for name in poly_names:
                for la in lmap[name]:
                    pts = (la - bx) / dx
                    plt.fill(pts[:, 1], pts[:, 0], c=poly_color)
            for la in lmap['road_divider']:
                pts = (la - bx) / dx
                plt.plot(pts[:, 1], pts[:, 0], c=road_divider_color)
            for la in lmap['lane_divider']:
                pts = (la - bx) / dx
                plt.plot(pts[:, 1], pts[:, 0], c=lane_divider_color)
    
            plt.xlim((self.X, 0))
            plt.ylim((0, self.Z))
    
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw')
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
    
            plt.close(fig)
            
            img_arr = rgba2rgb(img_arr)
            map_vis = torch.from_numpy(img_arr.astype(float) / 255.0)  # H, W, 3
            map_vis = map_vis.permute(2,0,1).float()                   # 3, H, W
            _, mH, mW = map_vis.shape
    
            # =========================================================
            num_vehicles = ptslist_.shape[0]
            if num_vehicles > 0:
                vehicle_masks = self.get_seg_bev(ptslist_)    # N, Z, X  
                
            else:
                # no vehicles in this sample
                vehicle_masks = torch.zeros((0, self.Z, self.X), dtype=torch.float32) # N, Z, X
    
            vehicle_masks = (vehicle_masks > 0).float()       # N, Z, X

            np.savez(cache_path, map_rgb=map_vis.numpy(), vehicle_masks=vehicle_masks.numpy())
        
        # ================= Extract Road Element Masks =============
        tolerance = 0.5
        target_color = torch.tensor(poly_color).view(3, 1, 1)
        diff = torch.abs(map_vis - target_color)
        poly_mask = torch.all(diff < tolerance, dim=0)

        target_color = torch.tensor(road_divider_color).view(3, 1, 1)
        diff = torch.abs(map_vis - target_color)
        road_divider_mask = torch.all(diff < tolerance, dim=0)

        target_color = torch.tensor(lane_divider_color).view(3, 1, 1)
        diff = torch.abs(map_vis - target_color)
        lane_divider_mask = torch.all(diff < tolerance, dim=0)

        # ================= Extract Vehicles =====================


        num_vehicles = vehicle_masks.shape[0]
        
        if num_vehicles > 0:
            vehicle_masks = F.interpolate(vehicle_masks.unsqueeze(0), (mH, mW)).squeeze(0)      # N, Z, X
            vehicle_masks = torch.flip(vehicle_masks, [1])  
    
            vehicle_masks_on_road = ((vehicle_masks == 1) & (poly_mask == 1)) | ((vehicle_masks == 1) & (road_divider_mask == 1)) | ((vehicle_masks == 1) & (lane_divider_mask == 1))
            vehicle_masks_on_road = vehicle_masks_on_road.float()
    
            vehicle_drop_rate = self.vehicle_drop_rate
            drop_mask = torch.rand(num_vehicles) < vehicle_drop_rate  # Randomly select masks to drop based on drop_rate
            vehicle_masks_on_road[drop_mask] = 0
    
            all_zero_masks = (vehicle_masks_on_road.sum(dim=(1, 2)) == 0)    
            num_all_zero_masks = all_zero_masks.sum().item()
            num_vehicles -= num_all_zero_masks

            seg_bev_on_road = (vehicle_masks_on_road[~all_zero_masks].sum(0) > 0).unsqueeze(0).float()  # 1, Z, X
        else: 
            seg_bev_on_road = torch.zeros((1, self.Z, self.X), dtype=torch.float32) # N, Z, X
            seg_bev_on_road = F.interpolate(seg_bev_on_road.unsqueeze(0), (mH, mW)).squeeze(0)      # N, Z, X

        
        prompt = '{} red rectangles on the road. city {}, center of the image is {} lane'.format(num_vehicles, city_name, direction) 
        if self.is_train and self.guidance_scale > 1.0 and random.random() < 0.1:
            prompt = ''
        
        # ================= Colorize MAP ===========================
        seg_bev_on_road_rgb = self.convert_to_rgb(seg_bev_on_road,  color=[1.0, 0.0, 0.0])   # 3, Z, X
        map_vis = map_vis * (1-seg_bev_on_road.float()) + seg_bev_on_road_rgb * seg_bev_on_road.float()                 # 3, Z, X
        
        assert map_vis.shape == (3,400,400), f'Invalid shape: {map_vis.shape}'

        rgb_tensor =  F.interpolate(map_vis.unsqueeze(0), (self.final_dim[0], self.final_dim[1])).squeeze(0)  # 3, Z, X

        # normalize to [-1, 1] from [0, 1]
        rgb_tensor = rgb_tensor * 2 - 1

        return  rgb_tensor, prompt
    

    def uv_to_color(self, uv_mask):
        u = uv_mask[0]      # Z, X
        v = uv_mask[1]      # Z, X

        size = u.shape[-1]

        angle = torch.atan2(v, u)
        hue = (angle + np.pi) / (2 * np.pi)

        color_mask = torch.zeros(3, size, size)

        for i in range(size):
            for j in range(size):
                if u[i,j] == 0 and v[i,j] == 0:
                    color_mask[:, i, j] = torch.tensor([1.,1.,1.])
                else:
                    color = plt.cm.hsv(hue[i, j].item())[:3] 
                    color_mask[:, i, j] = torch.tensor(color)

        return color_mask
    
    

    def convert_to_rgb(self, seg_bev, color):
        
        Z, X = seg_bev.shape[-2:]
        seg_bev = seg_bev.squeeze(0)            # Z, X
            
        seg_bev_rgb = torch.ones((3, Z, X), dtype=torch.float32)

        red_color = torch.tensor(color).view(3, 1)  
        
        vehicle_mask = (seg_bev == 1)  
        seg_bev_rgb[:, vehicle_mask] = red_color

        return seg_bev_rgb


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
        # in degrees
        ego_yaw_trans = Quaternion(egopose['rotation']).inverse.yaw_pitch_roll[0] * 180 / np.pi  
        
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        ptslist = []
        yawlist = []
        sizelist = []
        bbox2dlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if ('vehicle' not in inst['category_name']): # or ('cycle' in inst['category_name']):
                continue
            
            if int(inst['visibility_token']) == 1:
                #vislist.append(torch.tensor(0.0)) # invisible
                continue
            
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)
            
            pts = box.bottom_corners()[:2].T    # Bottom corners. First two face forward, last two face backwards.
            pts = np.round((pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
            
            pts[:, [1, 0]] = pts[:, [0, 1]]   # 4, 2 - switch x, y
            pts[:, 0] = self.X - 1 - pts[:, 0]       # horizontal flip
            
            if np.any(pts < 0) or np.any(pts > self.X):
                continue
            
            # NOTE: invisibility not important for DETR
            vislist.append(torch.tensor(1.0)) # visible
            
            center_pt = np.mean(pts,axis=0)  # 2
            
            ptslist.append(torch.from_numpy(pts))               

        if len(ptslist):
            vislist = torch.stack(vislist, dim=0)
            ptslist = torch.stack(ptslist, dim=0)
            # yawlist = torch.stack(yawlist, dim=0)
            # sizelist = torch.stack(sizelist, dim=0)
            # bbox2dlist = torch.stack(bbox2dlist, dim=0)
            # tidlist = torch.stack(tidlist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros((0))
            ptslist = torch.zeros((0, 4, 2))
            yawlist = torch.zeros((0))
            sizelist = torch.zeros((0, 2))
            bbox2dlist = torch.zeros((0, 5))
            # tidlist = torch.zeros((0))

        return lrtlist, boxlist, vislist, tidlist, ptslist, yawlist, sizelist, bbox2dlist
    

    
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
    
    