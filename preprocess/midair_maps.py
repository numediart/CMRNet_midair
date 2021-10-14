
# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")

import h5py
import numpy as np
import open3d as o3
import pykitti
import torch
from tqdm import tqdm

from utils import to_rotation_matrix

from PIL import Image

def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img

def convert_midair_depth_maps(depth_map):
    a = np.array([[1,0],[0,0]])
    sh = depth_map.shape
    f = sh[0]/2*np.ones(sh)
    focal = sh[0]/2

    a = np.tile(a, (int(focal), int(focal)))
    valid_indices = depth_map < np.max(depth_map.flatten())
    valid_indices = valid_indices & a == 1
    valid_indices = valid_indices.flatten()

    ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='xy')

    r_xy = depth_map/np.sqrt((xmap - f)**2 + (ymap - f)**2 + f**2)
    x_cam = r_xy*(xmap - f)
    y_cam = r_xy*(ymap - f)
    z_cam = r_xy*f
    x_cam = x_cam.flatten()
    y_cam = y_cam.flatten()
    z_cam = z_cam.flatten()
    return x_cam[valid_indices], y_cam[valid_indices], z_cam[valid_indices]

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', default='trajectory_5002',
                    help='sequence')
parser.add_argument('--device', default='cuda',
                    help='device')
parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
parser.add_argument('--start', default=0, help='Starting Frame')
parser.add_argument('--end', default=100000, help='End Frame')
#parser.add_argument('--map', default=None, help='Use map file')
#parser.add_argument('--kitti_folder', default='./KITTI/ODOMETRY', help='Folder of the KITTI dataset')
parser.add_argument('--midair_folder', default='/home/ambroise/dataset/MidAir/PLE_training/spring')

args = parser.parse_args()

sequence = args.sequence
#print("Sequence: ", sequence)

root_dir = args.midair_folder
depth_dir = os.path.join(root_dir, 'depth', sequence)
depth_list = os.listdir(depth_dir)
depth_list = [i for i in depth_list if '.PNG' in i]
depth_list = sorted(depth_list)

sensor_data = h5py.File(os.path.join(root_dir, 'sensor_records.hdf5'), 'r')
groundtruth = sensor_data[sequence]['groundtruth']
position = groundtruth['position']
attitude = groundtruth['attitude']
x = position[:,0]
y = position[:,1]
z = position[:,2]
q0 = attitude[:,0]
q1 = attitude[:,1]
q2 = attitude[:,2]
q3 = attitude[:,3]

pose_length = x.shape[0]
depth_length = len(depth_list)
poses = []

for i in range(0, pose_length, 4):
    T = torch.tensor([float(x[i]), float(y[i]), float(z[i])])
    R = torch.tensor([float(q0[i]), float(q1[i]), float(q2[i]), float(q3[i])])
    poses.append(to_rotation_matrix(R,T))

pc = o3.PointCloud()
#for i in range(int(args.start), int(depth_length)):
for i in range(int(args.start), int(args.end)):
    print('processing depth map {}'.format(depth_list[i]))
    depth_map = open_float16(os.path.join(depth_dir, depth_list[i]))
    z_b, y_b, x_b = convert_midair_depth_maps(depth_map)
    
    xyz = np.zeros((np.size(z_b), 4))
    xyz[:, 0] = x_b
    xyz[:, 1] = y_b
    xyz[:, 2] = z_b
    xyz[:, 3] = 1.

    RT = poses[i].numpy()
    xyz_rot_t = np.matmul(RT, xyz.T)
    xyz_rot_t = xyz_rot_t.T.copy()
    
    local_pc = o3.PointCloud()
    local_pc.points = o3.Vector3dVector(xyz_rot_t[:,:3])
    
    downpcd = o3.voxel_down_sample(local_pc, voxel_size = args.voxel_size)

    pc.points.extend(downpcd.points)
    
downpcd_full = o3.voxel_down_sample(pc, voxel_size= args.voxel_size)
downpcd, ind = o3.statistical_outlier_removal(downpcd_full, nb_neighbors=40, std_ratio=0.3)
o3.write_point_cloud(f'./map-{sequence}_{args.voxel_size}.pcd', downpcd_full)

voxelized = torch.tensor(downpcd.points, dtype=torch.float)
voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
voxelized = voxelized.t()
voxelized = voxelized.to(args.device)

left2right = torch.from_numpy(np.array([[1, 0, 0, 0],[0, 1, 0, -0.5], [0, 0, 1, 0], [0, 0, 0, 1]])).float().to(args.device)

if not os.path.exists(os.path.join(root_dir, 'sequences', sequence, f'local_maps_{args.voxel_size}')):
    os.makedirs(os.path.join(root_dir, 'sequences', sequence, f'local_maps_{args.voxel_size}'))

#for i in tqdm(range(int(args.start), int(depth_length))):
for i in tqdm(range(int(args.start), int(args.end))):
    pose = poses[i]
    pose = pose.to(args.device)
    pose = pose.inverse()

    local_map = voxelized.clone()
    #local_intensity = vox_intensity.clone()
    local_map = torch.mm(pose, local_map).t()
    #local_map = local_map[[2, 1, 0, 3],:]
    indexes = local_map[:, 1] > -25.
    indexes = indexes & (local_map[:, 1] < 25.)
    indexes = indexes & (local_map[:, 0] > -10.)
    indexes = indexes & (local_map[:, 0] < 100.)
    local_map = local_map[indexes]
    #local_intensity = local_intensity[:, indexes]

    local_map = torch.mm(left2right, local_map.t())
    #local_map = local_map[[2, 0, 1, 3], :]

    #pcd = o3.PointCloud()
    #pcd.points = o3.Vector3dVector(local_map[:,:3].numpy())
    #o3.write_point_cloud(f'{i:06d}.pcd', pcd)

    file = os.path.join(root_dir, 'sequences', sequence,
                        f'local_maps_{args.voxel_size}', f'{i:06d}.h5')
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)
        #hf.create_dataset('intensity', data=local_intensity.cpu().half(), compression='lzf', shuffle=True)



