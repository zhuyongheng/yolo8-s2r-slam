import os


filepath = '/home/zhu/dataset/KITTI/01/image_0'
file = open('list.txt', 'w')
filename = os.listdir(filepath)
filename.sort()
for n in filename:
    path = os.path.join(filepath, n)
    file.write(path + '\n')
file.close()