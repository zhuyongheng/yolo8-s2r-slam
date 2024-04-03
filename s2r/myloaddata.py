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


class MyDataset(data.Dataset):
    def __init__(self, filename, img_transform=None, joint_transform=None):
        self.img = []
        self.img_transform = img_transform
        self.joint_transform = joint_transform

        # 从list中分次读入图片 filename就是list的路径 以列表的形式存在self.img中
        f = open(filename, 'r')
        for data in f:
            data = data.strip('\n')
            self.img.append(data)

        # 加载数据集
        # fb = 718.856 * 386.145  # get focal_length * baseline

    def __len__(self):
        return len(self.img)


    # 这个方法在for循环启动的时候调用。利用它来读取数据
    def __getitem__(self, index):
        datafiles = self.img[index]
        image = Image.open(datafiles).convert('RGB')  # 读取图片
        # image.show()
        # cv2.waitKey(0)
        # image = np.array(image)
        # cv2.imshow("image2", image)
        # cv2.waitKey(0)

        # image =image.resize((1240, 372))
        image = self.joint_transform(image)     # image_aug
        image_to_tensor = self.img_transform(image)    # image to tensor

        return image_to_tensor
        # return



    # 这个方法在for循环启动的时候调用。利用它来读取数据
    # def __getitem__(self, index):
    #     datafiles = self.img[index]
    #     image = Image.open(datafiles).convert('RGB')  # 读取图片
    #     # image =image.resize((1240, 372))
    #     image = self.joint_transform(image)     # image_aug
    #     image_to_tensor = self.img_transform(image)    # image to tensor
    #
    #     return image_to_tensor


def create_test_dataloader(dataset, root, filename,
                           nThreads, loadSize):

    # 图像预处理
    joint_transform_list = [transform.MyRandomImgAugment(True, True, True, loadSize, dataset)]
    img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]

    joint_transform = Compose(joint_transform_list)
    # img_transform是将图像转换成tensor再做一个归一化处理
    img_transform = Compose(img_transform_list)

    # 这里是开始做数据集的加载
    test_data = MyDataset(filename=filename, img_transform=img_transform, joint_transform=joint_transform)

    loader = torch.utils.data.DataLoader(test_data,
                                batch_size=1,
                                shuffle=False,
                                # num_workers=int(nThreads),
                                num_workers=0,
                                pin_memory=True)
    return loader