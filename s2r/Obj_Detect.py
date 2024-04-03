import os

import cv2
import numpy
import torch
import random

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from collections import Counter
from torchvision.transforms import Compose, Normalize, ToTensor
import transform

from ultralytics import YOLO



import socket
import cv2 as cv
import torch
import argparse
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



def send_data(s: socket, image: np.ndarray):
    print('开始返回图片')
    img = image.flatten()
    len_data = img.shape[0]
    res = len_data % 65536
    for i in range(0, len_data - res, 65536):  # 从0开始每次传输1024位，传输次数为len_data / 1024
        s.send(img[i: i + 65536])
    if res != 0:
        s.send(img[-res:])

def recv_data(s: socket):
    data = b""
    while len(data) != 1397250:
    # while len(data) != 2764800:
    # while True:
        packet = s.recv(65536)
        print("/d",len(data))
        print("/d", len(packet))

        data += packet


    nparr = np.frombuffer(data, np.uint8)
    image = nparr.reshape(375, 1242, 3)
    # image = nparr.reshape(720, 1280, 3)


    print("接收完成")
    return image

def main():
    # global args
    args = parser.parse_args()

    # 接收图片
    img = recv_data(con)
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    # yolo检测
    # results = yolo8_model.predict(source=img, save=True, stream=True)
    results = yolo8_model.predict(source=img)
    for result in results:
        xyxy = result.boxes.xyxy.cpu().numpy().astype('uint16')  # box with xyxy format, (N, 4)
        conf = result.boxes.conf  # confidence score, (N, 1)
        result.boxes.cls.cpu().numpy()  # cls, (N, 1)
        img_detect = result.orig_img
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            score = "{:.2f}".format(conf[i])
            cv2.rectangle(img_detect, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img_detect, str(score), (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    send_data(con, img_detect)


if __name__ == '__main__':


    # 加载yolov8
    yolo8_model = YOLO("/home/zhu/ultralytics-main/yolov8m.pt")

    # 连接socket
    ADDRESS = ('127.0.0.1', 5000)
    # 创建一个socket连接
    tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpClient.bind(ADDRESS)
    # 连接服务器
    print("服务器连接中......")
    tcpClient.listen(1)
    con, address = tcpClient.accept()
    print("服务器连接成功！")


    while True:
        main()