
# -*- coding: utf-8 -*-

import os
import random
import shutil

# 设置文件路径和划分比例
root_path = "D:/dachuang/ultralytics/datasets/motobicycle/"
image_dir = "D:/dachuang/ultralytics/datasets/motobicycle/JPEGImages/train2/"
label_dir = "D:/dachuang/ultralytics/datasets/motobicycle/Annotations/train2/"
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 创建训练集、验证集和测试集目录
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("images/test", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)
os.makedirs("labels/test", exist_ok=True)

# 获取所有图像文件名
image_files = os.listdir(image_dir)
total_images = len(image_files)
random.shuffle(image_files)

# 计算划分数量
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count

# 划分训练集
train_images = image_files[:train_count]
for image_file in train_images:
    label_file = image_file[:image_file.rfind(".")] + ".txt"
    shutil.copy(os.path.join(image_dir, image_file), "images/train/")
    shutil.copy(os.path.join(label_dir, label_file), "labels/train/")

# 划分验证集
val_images = image_files[train_count:train_count+val_count]
for image_file in val_images:
    label_file = image_file[:image_file.rfind(".")] + ".txt"
    shutil.copy(os.path.join(image_dir, image_file), "images/val/")
    shutil.copy(os.path.join(label_dir, label_file), "labels/val/")

# 划分测试集
test_images = image_files[train_count+val_count:]
for image_file in test_images:
    label_file = image_file[:image_file.rfind(".")] + ".txt"
    shutil.copy(os.path.join(image_dir, image_file), "images/test/")
    shutil.copy(os.path.join(label_dir, label_file), "labels/test/")

# 生成训练集图片路径txt文件
with open("train.txt", "w") as file:
    file.write("\n".join([root_path + "images/train/" + image_file for image_file in train_images]))

# 生成验证集图片路径txt文件
with open("val.txt", "w") as file:
    file.write("\n".join([root_path + "images/val/" + image_file for image_file in val_images]))

# 生成测试集图片路径txt文件
with open("test.txt", "w") as file:
    file.write("\n".join([root_path + "images/test/" + image_file for image_file in test_images]))

print("数据划分完成！")
