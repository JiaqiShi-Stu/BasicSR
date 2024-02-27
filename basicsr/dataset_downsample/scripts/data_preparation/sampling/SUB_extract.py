import os
import random
import shutil

# 三个源文件夹路径
folder1 = '/data/sjq/srDataset/SUB_IMAGE/SD7K/test/HR'
folder2 = '/data/sjq/srDataset/SUB_IMAGE/SD7K/test/LR'
folder3 = '/data/sjq/srDataset/SUB_IMAGE/SD7K/test/SR'
# 目标文件夹路径
target_folder1 = '/data/sjq/srDataset/SUB_IMAGE/SD7K/val/HR'
target_folder2 = '/data/sjq/srDataset/SUB_IMAGE/SD7K/val/LR'
target_folder3 = '/data/sjq/srDataset/SUB_IMAGE/SD7K/val/SR'

# 获取三个文件夹中所有图片的名称
images1 = [file for file in os.listdir(folder1) if file.endswith('.png')]

# 随机抽取10张相同名字的图片
selected_images = random.sample(images1, 10)

# 创建目标文件夹
os.makedirs(target_folder1, exist_ok=True)
os.makedirs(target_folder2, exist_ok=True)
os.makedirs(target_folder3, exist_ok=True)

# 将选中的图片从三个文件夹中复制到目标文件夹
for image in selected_images:
    shutil.copy(os.path.join(folder1, image), target_folder1)
    shutil.copy(os.path.join(folder2, image), target_folder2)
    shutil.copy(os.path.join(folder3, image), target_folder3)