# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os
import socket
import cv2 as cv
import torch
import argparse
# import loaddata
import myloaddata
import matplotlib
import numpy as np
from utils import *
import matplotlib.cm
import torch.nn as nn
import DSAModules

import torch.nn.parallel
import matplotlib as mpl
from models import modules

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

import matplotlib.image as mpimg

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

plt.switch_backend('TkAgg')
plt.set_cmap("gray")
# plt.set_cmap("jet")

## =========================== Parameters =================
parser = argparse.ArgumentParser(description="S2R-DepthNet-Test.")
parser.add_argument('--dataset', type=str, default='KITTI', help='synthetic domain')                       # *******
parser.add_argument('--root',  type=str, default='', help='path to source dataset.')                        # *******
parser.add_argument('--filename', type=str, default='', help='stores data list, in syn_root')         # *******
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')                            # *******
parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')                     # *******
parser.add_argument('--loadSize', nargs='+', type=int, default=286, help='scale images to this size')        # *******
parser.add_argument('--out_dir', type=str, default="out", help="the path of log")
parser.add_argument('--Shared_Struct_Encoder_path', type=str, default="/home/zhu/s2r_model/outdoor/struct_encoder_vkitti.pth", help='the path of Shared_Struct_Encoder models')
parser.add_argument('--Struct_Decoder_path', type=str, default="/home/zhu/s2r_model/outdoor/struct_decoder_vkitti.pth", help='the path of Struct_Decoder models')
parser.add_argument('--DepthNet_path', type=str, default="/home/zhu/s2r_model/outdoor/depthnet_vkitti.pth", help='the path of DepthNet models')
parser.add_argument('--DSAModle_path', type=str, default='/home/zhu/s2r_model/outdoor/dsamodels_vkitti.pth', help='the path of DSAModle')


def save_test(handle, result1_log):
	'''
	This function save the test metrics in a given file.
	------
	handle: text file handle
	result_log: the metrics results, a 2D list
	'''
	abs_rel_1 = np.array(result1_log[0]).mean()
	sq_rel_1 = np.array(result1_log[1]).mean()
	rmse_1 = np.array(result1_log[2]).mean()
	rmse_log_1 = np.array(result1_log[3]).mean()
	a1_1 = np.array(result1_log[4]).mean()
	a2_1 = np.array(result1_log[5]).mean()
	a3_1 = np.array(result1_log[6]).mean()


	# write test result to test file by using handle
	handle.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n" \
			.format('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))

	handle.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n"\
			.format(abs_rel_1, sq_rel_1, rmse_1, rmse_log_1, a1_1, a2_1, a3_1))



def kitti_metrics_preprocess(pred, gt):
	'''
	This function do some preprocessing before metrics calculation
	- check zero value to avoid numerical problems;
	-
	Note that the input 'pred' and 'gt' are both 4D nparrays
	return the corresponding image pair 
	'''
	# squeeze the first and last idx(which is one in test processing)

	pred = np.squeeze(pred)
	gt = np.squeeze(gt)

	min_depth = 1e-3
	max_depth = 80
	pred[pred < min_depth] = min_depth
	pred[pred > max_depth] = max_depth

	mask = np.logical_and(gt > min_depth, gt < max_depth)
	gt_height, gt_width = gt.shape
	crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
				 0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
	crop_mask = np.zeros(mask.shape)
	crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
	mask = np.logical_and(mask, crop_mask)
	scalar = np.median(gt[mask])/np.median(pred[mask])
	pred[mask] *= scalar

	return pred[mask], gt[mask]



def kitti_compute_metrics(pred, gt):
	"""
	This function computes the metrics value on a pair of (pred, gt).
	Note that the input 'pred' and 'gt' are both nparrays
	Return a list of result float-values which correspond to MAE, MSE, RMSE, and a1, a2, a3
	"""
	# test image pre-processing 
	pred, gt = kitti_metrics_preprocess(pred, gt)

	## compute MSE and RMSE
	mse = ((gt - pred) ** 2).mean()
	rmse = np.sqrt(mse)
	
	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	# compute ap accuracy
	thresh = np.maximum((gt/pred), (pred/gt))
	a1 = (thresh < 1.25).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_rel = np.mean((np.abs(gt - pred) / gt))
	sq_rel = np.mean(((gt - pred) ** 2) / gt)
	#print("sq_rel:", sq_rel)

	return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

def nyu_compute_metrics(pred, gt):
	"""
	This function computes the metrics value on a pair of (pred, gt).
	Note that the input 'pred' and 'gt' are both nparrays
	Return a list of result float-values which correspond to MAE, MSE, RMSE, and a1, a2, a3
	"""
	# test image pre-processing 
	pred, gt = nyu_metrics_preprocess(pred, gt)
	#print("pred:", pred)
	#print("gt:", gt)
	#print("++++++++++++++++++++++++++++++++==")

	## compute MSE and RMSE
	mse = ((gt - pred) ** 2).mean()
	rmse = np.sqrt(mse)

	#print("rmse:", rmse)
	
	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	# compute ap accuracy
	thresh = np.maximum((gt/pred), (pred/gt))
	a1 = (thresh < 1.25).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_rel = np.mean((np.abs(gt - pred) / gt))
	sq_rel = np.mean(((gt - pred) ** 2) / gt)
	#print("sq_rel:", sq_rel)
	print(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

	return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]


def nyu_metrics_preprocess(pred, gt):
	'''
	This function do some preprocessing before metrics calculation
	- check zero value to avoid numerical problems;
	-
	Note that the input 'pred' and 'gt' are both 4D nparrays
	return the corresponding image pair 
	'''
	# squeeze the first and last idx(which is one in test processing)

	pred = np.squeeze(pred)
	gt = np.squeeze(gt)
	#print("gt1:", gt)


	min_depth = 1e-3
	max_depth = 8
	pred[pred < min_depth] = min_depth
	pred[pred > max_depth] = max_depth

	mask = np.logical_and(gt > min_depth, gt < max_depth)

	scalar = np.median(gt[mask])/np.median(pred[mask])
	pred[mask] *= scalar
		# gtiheight, gt_width = gt.shape

	#print("gt2:", gt[mask])
	return pred[mask], gt[mask]

def send_data(s: socket, image: np.ndarray):
    img = image.flatten()
    len_data = img.shape[0]
    res = len_data % 1024
    for i in range(0, len_data - res, 1024):  # 从0开始每次传输1024位，传输次数为len_data / 1024
        s.send(img[i: i + 1024])
    if res != 0:
        s.send(img[-res:])

def recv_data(s: socket):
    data = b""
    while len(data) != 465750:
    # while True:
        packet = s.recv(65536)
        print("/d",len(data))
        print("/d", len(packet))
        # if len(packet) == 0:
        #     break
        data += packet


    # print("1")
    nparr = np.frombuffer(data, np.uint8)
    # image = nparr.reshape(375, 1242, 3)

    image = nparr.reshape(375, 1242)
    print("接收完成")
    return image


def run():
	global args
	args = parser.parse_args()
	# =========================== DataLoader ===============================
	print("Loading the dataset ...")

	real_loader = myloaddata.create_test_dataloader(dataset=args.dataset,
													  root=args.root,
													filename=args.filename,
													  nThreads=args.nThreads,
													  loadSize=args.loadSize)

	# print(real_loader)
	print("Loading data set is complete!")
	print("=======================================================================================")
	print("Building models ...")


	# Define Shared Structure Encoder
	Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4,
												   input_dim=3, dim=64,
												   norm='in', activ='lrelu',
												   pad_type='reflect')

	# Define Structure Decoder
	Struct_Decoder = modules.Struct_Decoder()

	# Define Depth-specific Attention (DSA) module

	Attention_Model = DSAModules.drn_d_22(pretrained=True)
	DSAModle = DSAModules.AutoED(Attention_Model)

	# Define DepthNet
	DepthNet = modules.Depth_Net()
	init_weights(DepthNet, init_type='normal')

	Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()
	Struct_Decoder = torch.nn.DataParallel(Struct_Decoder).cuda()
	DSAModle = torch.nn.DataParallel(DSAModle).cuda()
	DepthNet = torch.nn.DataParallel(DepthNet).cuda()

	# Load models
	Shared_Struct_Encoder.load_state_dict(torch.load(args.Shared_Struct_Encoder_path))
	Struct_Decoder.load_state_dict(torch.load(args.Struct_Decoder_path))
	DSAModle.load_state_dict(torch.load(args.DSAModle_path))
	DepthNet.load_state_dict(torch.load(args.DepthNet_path))

	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)

	Shared_Struct_Encoder.eval()
	Struct_Decoder.eval()
	DSAModle.eval()
	DepthNet.eval()


	step = 0

	for i, real_batched in enumerate(real_loader):
		print("step:", step+1)
		image = real_batched

		# image = recv_data(con)

		image = torch.autograd.Variable(image).cuda()

		# predict
		struct_code = Shared_Struct_Encoder(image)
		structure_map = Struct_Decoder(struct_code)

		attention_map = DSAModle(image)
		depth_specific_structure = attention_map * structure_map

		pred_depth = DepthNet(depth_specific_structure)
		pred_depth = torch.nn.functional.interpolate(pred_depth[-1], size=[375, 1242], mode='bilinear',align_corners=True)

		pred_depth_np = np.squeeze(pred_depth.cpu().detach().numpy())

		pred_depth_np += 1.0
		pred_depth_np /= 2.0
		# pred_depth_np *= 80.0
		pred_depth_np *= 65535.0
		pred_depth_np = pred_depth_np.astype(np.uint16)

		# save_path = "./result/" + str(step + 1) + ".png"
		# cv.imwrite(save_path, pred_depth_np)
		# cv.destroyAllWindows()

		# send_data(con, pred_depth_np)
		step = step + 1




def main():
	global args
	args = parser.parse_args()
		# =========================== DataLoader ===============================
	print("Loading the dataset ...")




	real_loader = myloaddata.create_test_dataloader(dataset=args.dataset,
													  root=args.root,
													filename=args.filename,
													  nThreads=args.nThreads,
													  loadSize=args.loadSize)

	# print(real_loader)
	print("Loading data set is complete!")
	print("=======================================================================================")
	print("Building models ...")




	# Define Shared Structure Encoder
	Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4, 
												input_dim=3, dim=64, 
												norm='in', activ='lrelu', 
												pad_type='reflect')

	# Define Structure Decoder
	Struct_Decoder = modules.Struct_Decoder()

	# Define Depth-specific Attention (DSA) module

	Attention_Model = DSAModules.drn_d_22(pretrained=True)
	DSAModle = DSAModules.AutoED(Attention_Model)


	# Define DepthNet
	DepthNet = modules.Depth_Net()
	init_weights(DepthNet, init_type='normal')
	
	
	Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()
	Struct_Decoder = torch.nn.DataParallel(Struct_Decoder).cuda()
	DSAModle = torch.nn.DataParallel(DSAModle).cuda()
	DepthNet = torch.nn.DataParallel(DepthNet).cuda()	
	
	# Load models
	Shared_Struct_Encoder.load_state_dict(torch.load(args.Shared_Struct_Encoder_path))
	Struct_Decoder.load_state_dict(torch.load(args.Struct_Decoder_path))
	DSAModle.load_state_dict(torch.load(args.DSAModle_path))
	DepthNet.load_state_dict(torch.load(args.DepthNet_path))
	
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)


	Shared_Struct_Encoder.eval()
	Struct_Decoder.eval()
	DSAModle.eval()
	DepthNet.eval()

	step = 0

	for i, real_batched in enumerate(real_loader):
		print("step:", step+1)
		image = real_batched

		image = torch.autograd.Variable(image).cuda()

		# predict
		struct_code = Shared_Struct_Encoder(image)
		structure_map = Struct_Decoder(struct_code)

		attention_map = DSAModle(image)
		depth_specific_structure = attention_map * structure_map

		pred_depth = DepthNet(depth_specific_structure)
		pred_depth = torch.nn.functional.interpolate(pred_depth[-1], size=[720, 1280], mode='bilinear',align_corners=True)

		pred_depth_np = np.squeeze(pred_depth.cpu().detach().numpy())

		pred_depth_np += 1.0
		pred_depth_np /= 2.0
		# pred_depth_np *= 80.0
		pred_depth_np *= 65535.0
		pred_depth_np = pred_depth_np.astype(np.uint16)

		# cv.imshow("image", pred_depth_np)
		# cv.waitKey(0)
		save_path = "./result01/" + str(step + 1) + ".png"
		cv.imwrite(save_path, pred_depth_np)


		# send_data(con, pred_depth_np)

		step = step + 1

	
if __name__ == '__main__':

	# 连接socket
	# ADDRESS = ('127.0.0.1', 5678)
	# # 创建一个socket连接
	# tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# tcpClient.bind(ADDRESS)
	# # 连接服务器
	# print("服务器连接中......")
	# tcpClient.listen(1)
	# con, address = tcpClient.accept()
	# print("服务器连接成功！")

	t = True
	while t==True:
		try:
			main()
			# run()
			t = False
		except Exception as e:
			print(e)
			break
	# tcpClient.close()











