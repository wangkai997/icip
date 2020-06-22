# !/usr/bin/env python
# coding: utf-8
'''
@File    :   resnext101_finetune.py
@Time    :   2020/04/19 00:12:29
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to finetune ResNext101 model by Category or Subcategory

import argparse
import copy
import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm, trange
from resnest.torch import resnest50,resnest101,resnest200,resnest269
# net = resnest50(pretrained=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
random_seed=2020

def load_image(image_path):
    """ 
    load image data, convert image To Tenor
    """
    data_transforms = transforms.Compose([
        # transforms.Resize(224),
        transforms.Resize(416),
        # transforms.RandomCrop(224),
        transforms.CenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        return data_transforms(Image.open(image_path).convert("RGB"))
    except:
        print("There is no file : {}".format(image_path))


class Image_Data(Dataset):
    def __init__(self, dataset):
        self.image_list = dataset['image_path']
        self.FlickrId = dataset["FlickrId"]
        print("FlickrId:{} image:{} ".format(len(self.FlickrId),len(self.image_list)))

    def __len__(self):
        return len(self.FlickrId)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = load_image(image_path)
        return (self.FlickrId[index], image)


def initialize_model(model_name="resnext101",  use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    num_classes = 1000

    if model_name == "resnext101":
        """ ResNext101
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnest269":
        """ ResNest269
        """
        # model_ft = resnest269(pretrained=use_pretrained)
        model_ft = resnest269(pretrained=False)
        model_ft.load_state_dict(torch.load("/home/wangkai/pretrain_model/resnest269-0cc87c48.pth"))
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 416

    elif model_name == "resnext50":
        """ ResNext50
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet152":
        """ ResNet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def crete_dataset(image_root_path=None):
    """ 
    create image dataset
    """
    image_list=os.listdir(image_root_path)
    FlickrId=[item.split(".")[0] for item in image_list]
    image_path=[os.path.join(image_root_path, item) for item in image_list]
    data = {
        "FlickrId": FlickrId,
        "image_path":image_path
    }
    return data


def extract_feature(model, data_loader,image_feature_filepath="./feature.csv", use_gpu=True):

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_ftrs)
    torch.nn.init.eye_(model.fc.weight)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if use_gpu == True:
        model.to("cuda")

    model.eval()
    with open(image_feature_filepath, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        columns = ["FlickrId"] + ["ResNext101_image"+str(i+1) for i in range(2048)]
        writer.writerow(columns)
        for _, (FlickrIds, images) in tqdm(enumerate(data_loader)):
            images = images.to("cuda") if use_gpu else images
            features = model(images)
            features = features.to("cpu").data.numpy()
            FlickrIds = np.array(FlickrIds).reshape(-1, 1)
            writer.writerows(np.concatenate((FlickrIds, features), axis=1).tolist())


def main(args):

    data = crete_dataset(args.image_root_path)
    print("Initializing Datasets and Dataloaders...")
    dataset = Image_Data(data)
    data_loader = DataLoader(dataset=dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Start building model')
    # model, input_size = initialize_model(model_name=args.model,use_pretrained=args.use_pretrained)
    model = torch.load("/home/wangkai/SMP/checkpoint/ResNext101_best_subcategory.pth", map_location="cpu")
    for item in list(model.children())[:-3]:
        for param in item.parameters():
            param.requires_grad=False

    print("extract pretrained feature...")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to("cpu")

    extract_feature(model, data_loader=data_loader,image_feature_filepath=args.feature_path, use_gpu=args.use_gpu)
    print("Over! extract Pretrained feature.\n")
    print("The process is over")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract Image Feature")
    parser.add_argument("--image_root_path", type=str,
                        default="/home/wangkai/ICIP/data/test/test_imgs",
                        help="image root path")
    parser.add_argument("--feature_path", type=str,
                        default="/home/wangkai/ICIP/feature/test_feature/ResNext101_image_7693.csv",
                        help="image root path")
    parser.add_argument("--model", type=str,
                        default="resnext101",
                        choices=["resnext101", "resnext50", "resnet152", "resnest269"],
                        help="model name")
    parser.add_argument("--use_pretrained", type=bool, default=True,
                        help="use pretrain weight in ImageNet(default: True)")
    parser.add_argument("--use_gpu", type=bool, default=True,
                        help="will use gpu(default: True)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size(default: 64)")
    # parser.add_argument("--learning_rate", type=float,
    #                     default=0.0001,
    #                     help="learning rate (default: 0.001)")
    # parser.add_argument("--num_epochs", type=int, default=1,
    #                     help="number epochs(default: 10)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number workers(default: 8)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args=parse_arguments()
    main(args)

    pass
