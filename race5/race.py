import torch
import wandb

# wandb.login()

import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json 
import yaml
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import subprocess

json_file_path = 'F:/code/deeplearningfollowlimu/d2l-zh/pytorch/data/cowboyoutfits/train.json'
data = json.load(open(json_file_path, 'r'))

count = 0
labels = [87, 1034, 131, 318, 588]

for x in labels:
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['category_id'] == x:
            count += 1
    print(x, count)

#将bbox从coco转换成YOLO格式
def coco2yolo_bbox(img_width, img_height, bbox):
    dw = 1. / img_width
    dh = 1. / img_height
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

#由于annotation不连续，要生成一个map映射
cate_id_map = {}
num = 0
for cate in data['categories']:
    cate_id_map[cate['id']] = num
    num += 1

print(cate_id_map)

#转换格式
train_pd = pd.DataFrame(columns=['id', 'file_name', 'annotation',
                                 'box_count', 'object_count', 'object_id',
                                 'image_width', 'image_height'])

for i in tqdm(range(len(data['images']))):
    filename = data['images'][i]['file_name']
    image_width = data['images'][i]['width']
    image_height = data['images'][i]['height']
    image_id = data['images'][i]['id']

    annotation = []
    box_count = 0
    object_count = 0
    object_id = []

    for anno in data['annotations']:
        if anno['image_id'] == image_id:
            box_count += 1
            yolo_bbox = coco2yolo_bbox(image_width, image_height, 
                                      anno['bbox'])#bbox:[x, y, width, height]
            temp_annotation = '{} {} {} {} {}'.format(cate_id_map[anno['category_id']],
                                                      yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3])

            annotation.append(temp_annotation)

            if cate_id_map[anno['category_id']] not in object_id:
                object_count += 1
                object_id.append(cate_id_map[anno['category_id']])

    train_pd.loc[i] = image_id, filename, annotation, box_count, object_count, object_id, image_width, image_height

print(train_pd.sample(10))

#准备K折数据
NUM_FOLD = 5
Fold = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=233)

df_folds = train_pd.copy()

df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['object_count'].values.astype(str),
    df_folds['box_count'].apply(lambda x: f'_{x // 5}').values.astype(str))

for fold_number, (train_index, val_index) in enumerate(Fold.split(
    X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

print(train_pd['box_count'].value_counts())

print(df_folds.sample(5))

#看新类的分布
print(df_folds['stratify_group'].value_counts())

#查看fold分布数量
print(df_folds['fold'].value_counts())

#检查函数，检查我们分组是否合理
def list_class_number(fold_number):
    pd_tmp = df_folds.loc[df_folds['fold'] == fold_number]

    belt = 0
    sunglasses = 0
    boot = 0
    cowboy_hat = 0
    jacket = 0

    for i in range(len(pd_tmp)):
        for x in pd_tmp.iloc[i]['object_id']:
            if x == 0:
                belt += 1
            if x == 1:
                sunglasses += 1
            if x == 2:
                boot += 1
            if x == 3:
                cowboy_hat += 1
            if x == 4:
                jacket += 1
    print(f'fold_{fold_number}, belt:{belt}, sunglasses:{sunglasses}, boot:{boot}, cowboy_bat:{cowboy_hat}, jacket:{jacket}\n')
    print(pd_tmp['box_count'].value_counts())
    print(pd_tmp['object_count'].value_counts())

print(list_class_number(0))
print(list_class_number(1))

#准备数据文件夹们
NUM_FOLD = 5
df = df_folds.copy()

for fold in range(NUM_FOLD):
    print(fold)
    #准备训练和验证数据集
    train_df = df.loc[df.fold != fold].reset_index(drop=True)
    valid_df = df.loc[df.fold == fold].reset_index(drop=True)
    #创建文件夹
    os.makedirs(f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/images/train',
                exist_ok=True)
    os.makedirs(f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/images/valid', exist_ok=True)
    os.makedirs(f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/labels/train', exist_ok=True)
    os.makedirs(f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/labels/valid', exist_ok=True)

    #将图像和注释转移到创建的文件夹中
    for i in tqdm(range(len(train_df))):
        train_row = train_df.loc[i]
        train_name = train_row.file_name.split('.')[0]
        copyfile(f'F:/code/deeplearningfollowlimu/d2l-zh/pytorch/data/cowboyoutfits/images/{train_name}.jpg',
                 f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/images/train/{train_name}.jpg')
        yolo_txt_file = open(f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/labels/train/{train_name}.txt', 'w')
        for ann in train_row['annotation']:
            yolo_txt_file.write(f'{ann}\n')
    yolo_txt_file.close()

    for i in tqdm(range(len(valid_df))):
        valid_row = valid_df.loc[i]
        valid_name = valid_row.file_name.split('.')[0]
        copyfile(f'F:/code/deeplearningfollowlimu/d2l-zh/pytorch/data/cowboyoutfits/images/{valid_name}.jpg',
                 f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/images/valid/{valid_name}.jpg')
        yolo_txt_file = open(f'F:/code/deeplearningfollowlimu/race5/training/dataset_fold_{fold}/labels/valid/{valid_name}.txt', 'w')
        for ann in valid_row['annotation']:
            yolo_txt_file.write(f'{ann}\n')
    yolo_txt_file.close()

#创建yaml文件
for fold in range(NUM_FOLD):
    data_yaml = dict(
        train = f'../dataset_folds_{fold}/images/train/',
        val = f'../dataset_folds_{fold}/images/valid',
        nc = 5,
        names = ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket']
    )

    # we will make the file under the yolov5/data/ directory.
    with open(f'F:/code/deeplearningfollowlimu/race5/yolov5/data/data_folds_{fold}.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

#训练
IMG_SIZE = 640
BATCH_SIZE = 64
EPOCHS = 20
MODEL = 'yolo5s.pt'
name = f'{MODEL}_BS_{BATCH_SIZE}_EP_{EPOCHS}_fold_'



for fold in range(NUM_FOLD):
    print('FOLD NUMBER:', fold)
    command = ['python', 'race5/yolov5/train.py',
           '--img', str(IMG_SIZE),
           '--batch', str(BATCH_SIZE),
           '--epochs', str(EPOCHS),
           '--data', f'data_folds_{fold}.yaml',
           '--weights', MODEL,
           '--save-period', '1',
           '--project', 'F:/code/deeplearningfollowlimu/race5',
           '--name', f'{name}-{fold}',
           '--cache']
    print('``````````````````````````````````````````````````\n')

    if fold > 1:
        break
    
    subprocess.call(command)