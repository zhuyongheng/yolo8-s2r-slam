# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

img_path = '/home/zhu/dataset/player/'
img_list = os.listdir(img_path)
img_list.sort()
print('img_list: ', img_list)

with open('test.txt', 'w') as f:
    for img_name in img_list:
        f.write(img_path+img_name + '\n')