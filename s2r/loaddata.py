# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os

import cv2
import torch
import random
import transform
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from collections import Counter
from torchvision.transforms import Compose, Normalize, ToTensor






def get_dataset(root, data_file='train.list', 
				dataset='vkitti', phase='train',
				img_transform=None, 
				depth_transform=None,
				joint_transform=None
				):
	DEFINED_DATASET = {'KITTI'}
	assert dataset.upper() in DEFINED_DATASET
	print("name:", dataset.upper())
	name2obj = {'KITTI': KittiDataset}
	return name2obj[dataset.upper()](root=root, data_file=data_file, phase=phase,
									 img_transform=img_transform,
									 depth_transform=depth_transform,
									 joint_transform=joint_transform)


class MyKittiDataset01(data.Dataset):
	def __init__(self, root='./datasets', data_file='tgt_train.list', phase='train',
				 img_transform=None, joint_transform=None, depth_transform=None):
		self.root = root
		self.data_file = data_file
		self.files = []
		self.phase = phase
		self.img_transform = img_transform
		self.joint_transform = joint_transform
		self.depth_transform = depth_transform

		with open(self.data_file, 'r') as f:
			data_list = f.read().split('\n')
			for data in data_list:
				if len(data) == 0:
					continue
				data_info = data.split(' ')

				# files是一个存储了数据集文件名的列表
				self.files.append({
					"l_rgb": data_info[0],
					"r_rgb": data_info[1],
					"cam_intrin": data_info[2],
					"depth": data_info[3]
					})

	def __len__(self):
		return len(self.files)

	def read_data(self, datafiles):
		assert os.path.exists(os.path.join(self.root, datafiles['l_rgb'])), "Image does not exist"
		# 读左图
		l_rgb = Image.open(os.path.join(self.root, datafiles['l_rgb'])).convert('RGB')
		assert os.path.exists(os.path.join(self.root, datafiles['r_rgb'])), "Image does not exist"
		# 读右图
		r_rgb = Image.open(os.path.join(self.root, datafiles['r_rgb'])).convert('RGB')

		fb = 718.856 * 386.145   # get focal_length * baseline

		return l_rgb, r_rgb, fb

	def __getitem__(self, index):
		if index > len(self)-1:
			index = index % len(self)
		# datafiles是存储了左右图、相机内参路径的list
		datafiles = self.files[index]
		l_img, r_img, fb = self.read_data(datafiles)

		l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'test', fb))
		l_img = self.img_transform(l_img)
		r_img = self.img_transform(r_img)

		data = {}
		data['left_img'] = l_img
		data['right_img'] = r_img
		data['fb'] = fb
		return data


class KittiDataset(data.Dataset):
	def __init__(self, root='./datasets', data_file='tgt_train.list', phase='train',
				 img_transform=None, joint_transform=None, depth_transform=None):
		self.root = root
		self.data_file = data_file
		self.files = []
		self.phase = phase
		self.img_transform = img_transform
		self.joint_transform = joint_transform
		self.depth_transform = depth_transform

		with open(self.data_file, 'r') as f:
			data_list = f.read().split('\n')
			for data in data_list:
				if len(data) == 0:
					continue
				data_info = data.split(' ')

				# files是一个存储了数据集文件名的列表
				self.files.append({
					"l_rgb": data_info[0],
					"r_rgb": data_info[1],
					"cam_intrin": data_info[2],
					"depth": data_info[3]
				})

	def __len__(self):
		return len(self.files)

	def read_data(self, datafiles):
		assert os.path.exists(os.path.join(self.root, datafiles['l_rgb'])), "Image does not exist"
		# 读左图
		l_rgb = Image.open(os.path.join(self.root, datafiles['l_rgb'])).convert('RGB')
		w = l_rgb.size[0]
		h = l_rgb.size[1]
		assert os.path.exists(os.path.join(self.root, datafiles['r_rgb'])), "Image does not exist"
		# 读右图
		r_rgb = Image.open(os.path.join(self.root, datafiles['r_rgb'])).convert('RGB')

		# 这里实例化一个KITTI的对象kitti
		kitti = KITTI()
		assert os.path.exists(os.path.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
		# 通过kiiti计算相机的fb
		fb = kitti.get_fb(os.path.join(self.root, datafiles['cam_intrin']))  # get focal_length * baseline
		assert os.path.exists(os.path.join(self.root, datafiles['depth'])), "Depth does not exist"

		# depth, depth_interp = kitti.get_depth(os.path.join(self.root, datafiles['cam_intrin']),
		# 									  os.path.join(self.root, datafiles['depth']), [h, w], interp=True)
		# return l_rgb, r_rgb, fb, depth, depth_interp
		return l_rgb, r_rgb, fb

	def __getitem__(self, index):
		if self.phase == 'train':
			index = random.randint(0, len(self) - 1)
		if index > len(self) - 1:
			index = index % len(self)
		# datafiles是存储了左右图、相机内参路径的list
		datafiles = self.files[index]
		# l_img, r_img , fb, depth, depth_interp = self.read_data(datafiles)
		l_img, r_img, fb = self.read_data(datafiles)

		# if self.joint_transform is not None:
		# 	if self.phase == 'train':
		# 		l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'train', fb))
		# 	else:
		# 		l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'test', fb))
		# if self.img_transform is not None:
		# 	l_img = self.img_transform(l_img)
		# 	if r_img is not None:
		# 		r_img = self.img_transform(r_img)
		l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'test', fb))
		l_img = self.img_transform(l_img)
		r_img = self.img_transform(r_img)

		if self.phase == 'test':
			data = {}
			data['left_img'] = l_img
			data['right_img'] = r_img
			# data['depth'] = depth
			data['fb'] = fb
			# data['depth_interp'] = depth_interp
			return data


class DepthToTensor(object):
	def __call__(self, input):
		arr_input = np.array(input)
		tensors = torch.from_numpy(arr_input.reshape((1, arr_input.shape[0], arr_input.shape[1]))).float()
		return tensors


def creat_train_dataloader(dataset, root, data_file, batchsize, nThreads, 
						   no_flip, no_rotation, no_augment, loadSize):
	
	joint_transform_list = [transform.RandomImgAugment(no_flip, no_rotation, no_augment, loadSize, dataset)]

	img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]
	joint_transform = Compose(joint_transform_list)
	img_transform = Compose(img_transform_list)
	depth_transform = Compose([DepthToTensor()])

	dataset = get_dataset(root=root, data_file=data_file, phase='train',
						  dataset=dataset,
						  img_transform=img_transform, depth_transform=depth_transform,
						  joint_transform=joint_transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
										 shuffle=True, num_workers=int(nThreads),
										 pin_memory=True)
	return loader

def create_test_dataloader(dataset, root, data_file, batchsize, nThreads, loadSize):


	# 图像预处理
	joint_transform_list = [transform.RandomImgAugment(True, True, True, loadSize, dataset)]

	img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]
	joint_transform = Compose(joint_transform_list)
	# img_transform是将图像转换成tensor再做一个归一化处理
	img_transform = Compose(img_transform_list)
	depth_transform = Compose([DepthToTensor()])

	dataset = get_dataset(root=root, data_file=data_file, phase='test',
						dataset=dataset, img_transform=img_transform, depth_transform=depth_transform,
						joint_transform=joint_transform)

	loader = torch.utils.data.DataLoader(
								dataset,
								batch_size=1,
								shuffle=False,
								num_workers=int(nThreads),
								pin_memory=True)
	return loader



class KITTI:
	def read_calib_file(self, path):
		# taken from https://github.com/hunse/kitti
		float_chars = set("0123456789.e+- ")
		data = {}
		with open(path, 'r') as f:
			for line in f.readlines():
				key, value = line.split(':', 1)
				value = value.strip()
				data[key] = value
				if float_chars.issuperset(value):
					# try to cast to float array
					try:
						data[key] = np.array(list(map(float, value.split(' '))))
					except ValueError:
						# casting error: data[key] already eq.value, so pass
						pass
		return data

	def get_fb(self, calib_dir, cam=2):
		cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
		P2_rect = cam2cam['P_rect_02'].reshape(3, 4)   # Projection matrix of the left camera
		P3_rect = cam2cam['P_rect_03'].reshape(3, 4)   # Projection matrix of the right camera

		# cam 2 is left of cam 0 -6cm
		# cam 3 is to the right +54cm

		b2 = P2_rect[0, 3] / -P2_rect[0,0]       # offset of cam 2 relative to cam0
		b3 = P3_rect[0, 3] / -P3_rect[0,0] 		 # offset of cam 3 relative to cam0

		baseline = b3 - b2

		if cam == 2:
			focal_length = P2_rect[0, 0]         # focal_length of cam 2
		elif cam == 3:
			focal_length = P3_rect[0, 0]         # focal_length of cam 3
		return focal_length * baseline

	def load_velodyne_points(self, file_name):
		# adapted from https://github.com/hunse/kitti
		points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
		points[:, 3] = 1.0
		return points

	def lin_interp(self, shape, xyd):
		# taken from https://github.com/hunse/kitti
		from scipy.interpolate import LinearNDInterpolator
		## m=h, n=w xyd
		m, n = shape
		ij, d = xyd[:, 1::-1], xyd[:, 2]
		f = LinearNDInterpolator(ij, d, fill_value=0)
		# h, w
		J, I = np.meshgrid(np.arange(n), np.arange(m))
		IJ = np.vstack([I.flatten(), J.flatten()]).T
		disparity = f(IJ).reshape(shape)
		return disparity

	def sub2ind(self, metrixSize, rowSub, colSub):
		# m=h, n=w
		# rowsub y
		# colsub x
		m, n = metrixSize

		return rowSub * (n-1) + colSub - 1  # num 


	def get_depth(self, calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
		# load calibration files
		cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
		velo2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
		velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
		velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))   # Projection matrix of Point cloud to cam

		# compute projection matrix velodyne --> image plane
		R_cam2rect = np.eye(4)
		R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3, 3) # Corrected rotation matrix for camera 0 to camera 0
		P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)    # Projection matrix of the left camera
		P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

		# load velodyne points and remove all behind image plane (approximation)
		# each row of the velodyne data is forward, left, up, reflectance
		velo = self.load_velodyne_points(velo_file_name)
		velo = velo[velo[:, 0]>=0, :]  # remove all behind image plane

		# project the points to camera
		velo_pts_im = np.dot(P_velo2im, velo.T).T
		velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] #homogenous --> not homogenous

		if vel_depth:
			velo_pts_im[:, 2] = velo[:, 0]

		# check is in bounds
		# use minus 1 to get the exact same value as KITTI matlab code

		velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
		velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
		val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
		val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1]<im_shape[0])
		velo_pts_im = velo_pts_im[val_inds, :]

		# project to image
		depth = np.zeros((im_shape))   # h, w
		depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

		# find the duplicate points and choose the closest depth
		# depth_shape = (h, w)   velo_pts_im[:, 1] y  velo_pts_im[:, 0] x
		inds = self.sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
		dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
		for dd in dupe_inds:
			pts = np.where(inds==dd)[0]
			x_loc = int(velo_pts_im[pts[0], 0])   # x
			y_loc = int(velo_pts_im[pts[0], 1])   # y
			depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
		depth[depth<0] = 0

		if interp:
			# interpolate the depth map to fill in holes
			depth_interp = self.lin_interp(im_shape, velo_pts_im)
			return depth, depth_interp
		else:
			return depth