import csv
import os 
from math import radians

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from camera_model import CameraModel
from utils import invert_pose, rotate_forward

def get_calib_midair():
    return torch.tensor([1024.0, 1024.0, 512.0, 512.0])

class DatasetVisibilityMidairSingle(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, maps_folder='local_maps_0.1', 
                use_reflectance=False, max_t=2., max_r=10, split='test', device='cpu', test_sequence='trajectory_5001'):
        super(DatasetVisibilityMidairSingle, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder=maps_folder
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform 
        self.split = split 
        self.GTs_R = {}
        self.GTs_T = {}

        self.all_files = []
        self.model = CameraModel()
        self.model.focal_length = [1024, 1024]
        self.model.principal_point = [512, 512]

        sensor_data = h5py.File(os.path.join(self.root_dir, 'sensor_records.hdf5'), 'r')
        #depth_dir = os.path.join(root_dir, 'depth')
        for trajectory in ['trajectory_5000', 'trajectory_5001', 'trajectory_5002', 'trajectory_5003', 'trajectory_5004', 'trajectory_5005', 'trajectory_5006']:
            groundtruth = sensor_data[trajectory]['groundtruth']
            position = groundtruth['position']
            attitude = groundtruth['attitude']
            self.GTs_R[trajectory] = []
            self.GTs_T[trajectory] = []
            x = position[:,0]
            pose_length = x.shape[0]
            depth_path = os.path.join(self.root_dir, 'depth', trajectory)
            depth_list = os.listdir(depth_path)
            depth_list = [trajectory + '/' + i for i in depth_list if '.PNG' in i]
            depth_list = sorted(depth_list)
            self.all_files = self.all_files + depth_list
            #self.all_files.append(depth_list)
            for i in range(0, pose_length, 4):
                GT_R = np.array([attitude[i,0], attitude[i,1], attitude[i,2], attitude[i,3]]) 
                GT_T = np.array([position[i,0], position[i,1], position[i,2]])
                self.GTs_R[trajectory].append(GT_R)
                self.GTs_T[trajectory].append(GT_T)

        self.test_RT = []
        if split == 'test':
            for i in range(len(self.all_files)):
                rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                transl_x = np.random.uniform(-max_t, max_t)
                transl_y = np.random.uniform(-max_t, max_t)
                transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                self.test_RT.append([i, transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])
    
    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)        

    def __getitem__(self, idx):
        item = self.all_files[idx]
        #print(item)
        run = str(item.split('/')[0])
        timestamp = str(item.split('/')[1])
        timestamp_without_ext = str(timestamp.split('.')[0])
        img_path = os.path.join(self.root_dir, 'color_right', run, timestamp_without_ext+'.JPEG')
        pc_path = os.path.join(self.root_dir, 'sequences', run, self.maps_folder, timestamp_without_ext+'.h5')

        try:
            with h5py.File(pc_path, 'r') as hf:
                pc = hf['PC'][:]
                if self.use_reflectance:
                    reflectance = hf['intensity'][:]
                    reflectance = torch.from_numpy(reflectance).float()
        except Exception as e:
            print(f'File Broken: {pc_path}')
            raise e

        pc_in = torch.from_numpy(pc.astype(np.float32))#.float()
        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[1, :] *= -1

        img = Image.open(img_path)
        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split != 'test':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.test_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = get_calib_midair()
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if not self.use_reflectance:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'idx': run, 'rgb_name': timestamp_without_ext}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'reflectance': reflectance, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'idx': run, 'rgb_name': timestamp_without_ext}

        return sample
            

            


