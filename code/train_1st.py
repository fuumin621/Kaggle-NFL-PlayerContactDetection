#%%
import argparse
import os

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config_path" , type=str, required=True)
parser.add_argument("--debug" , type=bool, required=False,default=False)
config_path = parser.parse_args().config_path
DEBUG = parser.parse_args().debug
with open(config_path, 'r') as f:
    CFG = yaml.safe_load(f)

# %%
exp_id =  config_path.split("/")[-1].split(".")[0]
if DEBUG:
    exp_id = "debug_" +exp_id
    CFG["train_bs"] = 4
    CFG["valid_bs"] = 4
# %%
save_dir = "./../model/" + exp_id.split(".")[0]
os.makedirs(save_dir, exist_ok=True)
save_dir

import copy
import gc
import glob
import math
# %%
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from functools import lru_cache

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from albumentations.pytorch import ToTensorV2
from numba import jit
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from timm.models.resnet import Bottleneck
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from turbojpeg import (TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJFLAG_PROGRESSIVE,
                       TJPF_GRAY, TJSAMP_GRAY, TurboJPEG)

import wandb

warnings.filterwarnings("ignore")
jpeg = TurboJPEG()

# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.OneOf([
        A.MotionBlur((3,21), p=1.0),
        A.Blur(p=1.0),
        A.GaussianBlur(p=1.0),
    ], p=0.5),
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])

valid_aug = A.Compose([
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])

# %%
feature_cols = [
    'distance',
    'distance_1',
    'distance_2',
    'speed_1',
    'speed_2',
    'acceleration_1',
    'acceleration_2',
    'same_team',
    'different_team',
    'G_flug',
]

input_path = "./../data/"
frame_path = "./../data/frames/train/"

# %%
with open(input_path + "video2helmets.pkl", "rb") as tf:
    video2helmets = pickle.load(tf)
    
with open(input_path + "video2frames.pkl", "rb") as tf:
    video2frames = pickle.load(tf)
    
with open(input_path + f"train_{CFG['num_channels']}-{CFG['step_train']}_dist2_std_sg5folds.pkl", "rb") as tf:
    train_list = pickle.load(tf)

# %%
class NFLDataset(Dataset):
    def __init__(self, data, aug=valid_aug, mode='train'):
        self.data = data
        self.aug  = aug
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        df = self.data[idx]

        frames      = df.frame.values
        contact_ids = df.contact_id.astype("str").to_list()
        contact_ids = [",".join(contact_ids)]

        game_play   = df.iloc[0]["game_play"]

        players = []
        for p in [df.iloc[0]["nfl_player_id_1"], df.iloc[0]["nfl_player_id_2"]]:
            if p == 'G':
                players.append(p)
            else:
                players.append(int(p))

        tmp_drop_rate = np.random.uniform()
        drop_rate = {
            'Endzone':tmp_drop_rate,
            'Sideline':1-tmp_drop_rate,
        }
                
        imgs      = []
        bbox_imgs = []
        for view in ['Endzone', 'Sideline']:
            video = game_play + f'_{view}.mp4'

            tmp = video2helmets[video]
            tmp = tmp[tmp['frame'].between(frames[0], frames[-1])]
            tmp = tmp[tmp.nfl_player_id.isin(players)]

            # 各フレームで2playerの平均bboxを作成
            bboxes    = tmp.groupby('frame')[['left','width','top','height']].mean()
            bboxes_p1 = tmp[tmp["nfl_player_id"]==players[0]][["frame",'left','width','top','height']].set_index("frame")
            if "G" not in players:
                bboxes_p2 = tmp[tmp["nfl_player_id"]==players[1]][["frame",'left','width','top','height']].set_index("frame")
                
            view_img      = []
            view_bbox_img = []
            for i, f in enumerate(frames):
                img_new      = np.zeros((CFG["img_size"], CFG["img_size"], 2), dtype=np.uint8)
                bbox_img_new = np.zeros((CFG["img_size"], CFG["img_size"], 2), dtype=np.uint8)

                # if flag  and f <= video2frames[video]:
                if (f in bboxes.index) and (f <= video2frames[video]):
                    if self.mode == "train" and np.random.uniform() < CFG["frame_drop_out"]:
                        pass
                    else:
                        # TODO 同じ処理なので関数化
                        if os.path.isfile(f'{frame_path}{video}_{f-1:04d}.jpg'):
                            path_a = str(f'{frame_path}{video}_{f-1:04d}.jpg')
                        else:
                            path_a = str(f'{frame_path}{video}_{f:04d}.jpg')
                            
                        if os.path.isfile(f'{frame_path}{video}_{f+1:04d}.jpg'):
                            path_b = str(f'{frame_path}{video}_{f+1:04d}.jpg')
                        else:
                            path_b = str(f'{frame_path}{video}_{f:04d}.jpg')
                        
                        in_file_a = open(path_a,"rb")
                        img_a = jpeg.decode(in_file_a.read(), pixel_format=TJPF_GRAY)[:, :, 0]
                        in_file_a.close()
                            
                        in_file_b = open(path_b,"rb")
                        img_b = jpeg.decode(in_file_b.read(), pixel_format=TJPF_GRAY)[:, :, 0]
                        in_file_b.close()
                        
                        img = np.stack([img_b, img_a]).transpose(1,2,0)
                        
                        x , w , y , h  = bboxes.loc[f][['left','width','top','height']]
                        wh = int(max(w, h) *  3)

                        x_min = max(0, int(x+w/2)-wh)
                        x_max = min(img.shape[1] ,int(x+w/2)+wh)
                        y_min = max(0, int(y+h/2)-wh)
                        y_max = min(img.shape[0], int(y+h/2)+wh)

                        img       = img[y_min:y_max,x_min:x_max,:].copy()
                        
                        bbox_img  = np.zeros(img.shape, dtype=np.uint8)
                        bh, bw, _ = bbox_img.shape

                        if f in bboxes_p1.index:
                            x1, w1, y1, h1 = bboxes_p1.loc[f][['left','width','top','height']]
                            x1 = max(0, x1-x_min)
                            y1 = max(0, y1-y_min)
                            bbox_img[int(y1):min(bh,int(y1+h1)), int(x1):min(bw,int(x1+w1)), :] += 1
                        
                        if "G" not in players:
                            if f in bboxes_p2.index:
                                x2, w2, y2, h2 = bboxes_p2.loc[f][['left','width','top','height']]
                                x2 = max(0, x2-x_min)
                                y2 = max(0, y2-y_min)
                                bbox_img[int(y2):min(bh,int(y2+h2)), int(x2):min(bw,int(x2+w2)), :] += 1
                                
                        try:
                            img      = cv2.resize(img, (CFG["img_size"], CFG["img_size"]))
                            bbox_img = cv2.resize(bbox_img, (CFG["img_size"], CFG["img_size"]))
                        except:
                            print(img.shape)
                            print(x_min, x_max, y_min, y_max)
                        img_new[:img.shape[0], :img.shape[1]] = img
                        bbox_img_new[:bbox_img.shape[0], :bbox_img.shape[1]] = bbox_img

                view_img.append(img_new)
                view_bbox_img.append(bbox_img_new)
            imgs.append(np.array(view_img))
            bbox_imgs.append(np.array(view_bbox_img))
            
        feature = df[feature_cols]
        feature = feature.fillna(-1).values
        feature = np.float32(feature)
        
        if self.mode == "train":
            feature[random.sample(list(range(len(feature))), int(len(feature) * CFG["feature_drop_out"])), :] = 0
        
        label   = df["contact"].values
        label   = label.T.reshape(-1)
        label   = np.float32(label)
        
        # padding mask用
        padding_mask   = (df["contact_id"] == "padding_id").values
        
        # 距離が2以下かどうか、後で推論結果をmaskするために使用
        mask           = df["distance_below_2"].values | df["G_flug"].values
        
        end_img        = imgs[0].transpose(0,3,1,2).reshape(-1, CFG["img_size"],CFG["img_size"]).transpose(1, 2, 0) 
        side_img       = imgs[1].transpose(0,3,1,2).reshape(-1, CFG["img_size"],CFG["img_size"]).transpose(1, 2, 0)
        end_bbox_imgs  = bbox_imgs[0].transpose(0,3,1,2).reshape(-1, CFG["img_size"],CFG["img_size"]).transpose(1, 2, 0) 
        side_bbox_imgs = bbox_imgs[1].transpose(0,3,1,2).reshape(-1, CFG["img_size"],CFG["img_size"]).transpose(1, 2, 0) 
        
        end_data       = self.aug(image=end_img, mask=end_bbox_imgs)
        side_data      = self.aug(image=side_img, mask=side_bbox_imgs)

        end_img        = torch.concat([end_data["image"].view(-1,2,CFG["img_size"],CFG["img_size"]), end_data["mask"].permute(2,0,1).view(-1,2,CFG["img_size"],CFG["img_size"])[:,1,:,:].unsqueeze(1)],dim=1)
        side_img       = torch.concat([side_data["image"].view(-1,2,CFG["img_size"],CFG["img_size"]), side_data["mask"].permute(2,0,1).view(-1,2,CFG["img_size"],CFG["img_size"])[:,1,:,:].unsqueeze(1)],dim=1)

        return end_img, side_img, feature, label, mask, contact_ids, padding_mask

# %%
end_img, side_img, feature, label, mask, contact_ids, padding_mask = NFLDataset(train_list[0], train_aug, 'train')[2]

plt.figure(figsize=(16,16))

for i in range(end_img.shape[0]):
    plt.subplot(CFG["num_channels"]//4,min(10,CFG["num_channels"]),2*i+1)
    plt.imshow(end_img[:, :1, :, :].permute(0,2,3,1)[i,:,:,:])
    plt.subplot(CFG["num_channels"]//4,min(10,CFG["num_channels"]),2*i+2)
    plt.imshow(end_img[:, 2, :, :].squeeze(1).permute(1,2,0)[:,:,i])
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)

# %%
plt.figure(figsize=(16,16))

for i in range(side_img.shape[0]):
    plt.subplot(CFG["num_channels"]//4,min(10,CFG["num_channels"]),2*i+1)
    plt.imshow(side_img[:, :1, :, :].permute(0,2,3,1)[i,:,:,:])
    plt.subplot(CFG["num_channels"]//4,min(10,CFG["num_channels"]),2*i+2)
    plt.imshow(side_img[:, 2, :, :].squeeze(1).permute(1,2,0)[:,:,i])
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)

# %%

# %%


class TemporalShift(nn.Module):
    def __init__(self, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,stride=1, downsample=None,n_segment=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.tsm = TemporalShift(n_segment=n_segment, n_div=8, inplace=False)

    def forward(self, x):
        identity = x

        out = self.tsm(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,n_segment=32):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.tsm = TemporalShift(n_segment=n_segment, n_div=8, inplace=False)

    def forward(self, x):
        identity = x

        out = self.tsm(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,n_segment=32):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],n_segment=n_segment)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,n_segment=n_segment)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,n_segment=n_segment)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,n_segment=n_segment)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,n_segment=32):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,n_segment=n_segment))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,n_segment=n_segment))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained,n_segment, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],n_segment=n_segment, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained,n_segment, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],n_segment=n_segment, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained,n_segment, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],n_segment=n_segment, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained,n_segment, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],n_segment=n_segment, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained,n_segment, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],n_segment=n_segment, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class TemporalBottleneck(nn.Module):
    def __init__(self, block, nsgm):
        super(TemporalBottleneck, self).__init__()
        self.tsm = TemporalShift(n_segment=nsgm, n_div=8, inplace=False)
        self.block = block

    def forward(self, x):
        x = self.tsm(x)
        x = self.block(x)
        return x
    
class TemporalShiftBlock(nn.Module):
    def __init__(self, block, nsgm):
        super(TemporalShiftBlock, self).__init__()
        self.tsm = TemporalShift(n_segment=nsgm, n_div=8, inplace=False)
        self.block = block

    def forward(self, x):
        x = self.block(x)
        x = self.tsm(x)
        return x

class NFLModel(nn.Module):
    def __init__(self, backbone, n_segment):
        super(NFLModel, self).__init__()
        
        if "resnet18" == backbone:
            model = resnet18(pretrained=True,n_segment=n_segment)
            model.fc = nn.Linear(model.fc.in_features, 128)
            self.backbone = model
        elif "resnet34"  == backbone:
            model = resnet34(pretrained=True,n_segment=n_segment)
            model.fc = nn.Linear(model.fc.in_features, 128)
            self.backbone = model
        elif "resnet50"  == backbone:
            model = resnet50(pretrained=True,n_segment=n_segment)
            model.fc = nn.Linear(model.fc.in_features, 128)
            self.backbone = model
        elif "resnext50"  == backbone:
            model = timm.create_model("resnext50d_32x4d", pretrained=True, in_chans=3, num_classes=128)
            self.backbone = model
            self.rep_tmp_bottleneck(n_segment)
        elif "effnet-v2-b0"  == backbone:
            model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, in_chans=3, num_classes=128)
            for i, block in enumerate(model.blocks):
                model.blocks[i] = TemporalShiftBlock(block, n_segment)
            self.backbone = model
        elif "effnet-b0"  == backbone:
            model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, in_chans=3, num_classes=128)
            for i, block in enumerate(model.blocks):
                model.blocks[i] = TemporalShiftBlock(block, n_segment)                
            self.backbone = model
        elif "effnet-b1"  == backbone:
            model = timm.create_model('tf_efficientnet_b1_ns', pretrained=True, in_chans=3, num_classes=128)
            for i, block in enumerate(model.blocks):
                model.blocks[i] = TemporalShiftBlock(block, n_segment)                
            self.backbone = model
        elif "effnet-b2"  == backbone:
            model = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, in_chans=3, num_classes=128)
            for i, block in enumerate(model.blocks):
                model.blocks[i] = TemporalShiftBlock(block, n_segment)                
            self.backbone = model
        
        self.mlp = nn.Sequential(
            nn.Linear(10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm1 = nn.LSTM(128*3, 128, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(128*2, 128, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128*2, 1)
        
    def rep_tmp_bottleneck(self, n_segment):
        for i in range(len(self.backbone.layer1)):
            self.backbone.layer1[i] = TemporalBottleneck(self.backbone.layer1[i], n_segment)

        for i in range(len(self.backbone.layer2)):
            self.backbone.layer2[i] = TemporalBottleneck(self.backbone.layer2[i], n_segment)

        for i in range(len(self.backbone.layer3)):
            self.backbone.layer3[i] = TemporalBottleneck(self.backbone.layer3[i], n_segment)

        for i in range(len(self.backbone.layer4)):
            self.backbone.layer4[i] = TemporalBottleneck(self.backbone.layer4[i], n_segment)

    def forward(self, end_img, side_img, feature):
        b, t, c, h, w = end_img.shape
        end_img = end_img.view(-1, 3, h, w)
        side_img = side_img.view(-1, 3, h, w)
        y1 = self.backbone(end_img).view(b, t, -1)
        y2 = self.backbone(side_img).view(b, t, -1)
        y3 = self.mlp(feature)
        y = torch.concat([y1,y2,y3], 2)
        y, _ = self.lstm1(y)
        y, _ = self.lstm2(y)
        y = self.fc(y)
        y = y.reshape(-1)
        
        return y

# %%
model = NFLModel(backbone = CFG["backbone"],n_segment=CFG["num_channels"])
x1  = torch.randn(4,CFG["num_channels"], 3,128,128)
x2  = torch.randn(4,CFG["num_channels"], 3,128,128)
y   = torch.randn(4, CFG["num_channels"], len(feature_cols))
out = model(x1, x2, y)
print(out.shape)

# %%



@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

@jit
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        y_prob.sort()
        plt.plot(y_prob, mccs) 
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc

# %%
def criterion(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true)

# %%
def train_one_epoch(model, optimizer, scheduler, dataloader, epoch):
    global best_loss
    global best_mcc
    
    model.train()
    scaler = GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (img1,img2, feature, label, _, _, _) in pbar:
        img1 = img1.to(device, dtype=torch.float)
        img2 = img2.to(device, dtype=torch.float)
        feature  = feature.to(device, dtype=torch.float)
        label  = label.to(device, dtype=torch.float)
        label = label.view(-1)

        batch_size = img1.size(0)
        
        with autocast(enabled=True):
            output = model(img1, img2, feature).view(-1)
            loss   = criterion(output, label)
            
        scaler.scale(loss).backward()
    
        scaler.step(optimizer)
        scaler.update()

        # zero the parameter gradients
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        
        if step == len(dataloader)//2:
                valid_loss, valid_mcc = valid_one_epoch(model, dataloader=valid_loader)

                print(f"train loss : {epoch_loss}, valid loss : {valid_loss}, mcc : {valid_mcc}")
                model.train()
                run.log({'mcc': valid_mcc, 'valid loss': valid_loss, "train loss": epoch_loss, "lr":current_lr})

                if valid_loss <= best_loss:
                    print(f"Valid Loss Improved ({best_loss:0.4f} ---> {valid_loss:0.4f})")
                    best_loss    = valid_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    PATH = os.path.join(save_dir, f"best_loss-{fold:02d}.bin")
                    torch.save(model.state_dict(), PATH)
                    print(f"Model Saved")
                if valid_mcc >= best_mcc:
                    print(f"Valid MCC Improved ({best_mcc:0.4f} ---> {valid_mcc:0.4f})")
                    best_mcc    = valid_mcc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    PATH = os.path.join(save_dir, f"best_mcc-{fold:02d}.bin")
                    torch.save(model.state_dict(), PATH)
                    print(f"Model Saved")
        
        
    valid_loss, valid_mcc = valid_one_epoch(model, dataloader=valid_loader)

    print(f"train loss : {epoch_loss}, valid loss : {valid_loss}, mcc : {valid_mcc}")
    run.log({'mcc': valid_mcc, 'valid loss': valid_loss, "train loss": epoch_loss, "lr":current_lr})

    if valid_loss <= best_loss:
        print(f"Valid Loss Improved ({best_loss:0.4f} ---> {valid_loss:0.4f})")
        best_loss    = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        PATH = os.path.join(save_dir, f"best_loss-{fold:02d}.bin")
        torch.save(model.state_dict(), PATH)
        print(f"Model Saved")
    if valid_mcc >= best_mcc:
        print(f"Valid MCC Improved ({best_mcc:0.4f} ---> {valid_mcc:0.4f})")
        best_mcc    = valid_mcc
        best_model_wts = copy.deepcopy(model.state_dict())
        PATH = os.path.join(save_dir, f"best_mcc-{fold:02d}.bin")
        torch.save(model.state_dict(), PATH)
        print(f"Model Saved")
        
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, current_lr

# %%
@torch.no_grad()
def valid_one_epoch(model, dataloader, show=False):    
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    labels = defaultdict(int)
    preds = defaultdict(list)

    for step, (img1,img2, feature, label, mask, contact_id, padding_mask) in enumerate(dataloader):
        img1 = img1.to(device, dtype=torch.float)
        img2 = img2.to(device, dtype=torch.float)
        feature  = feature.to(device, dtype=torch.float)
        label  = label.to(device, dtype=torch.float)
        label = label.view(-1)
        mask  = mask.to(device, dtype=torch.bool)
        mask = mask.view(-1)
        padding_mask  = padding_mask.to(device, dtype=torch.bool)
        padding_mask = padding_mask.view(-1)
        
        batch_size = img1.size(0)
        
        output = model(img1, img2, feature).view(-1)
        # 評価はdistance < 2のみで行う
        mask_all = mask | padding_mask
        output = output[mask_all]
        label = label[mask_all]
        
        # batch * num_channels
        contact_id = ",".join(contact_id[0])
        contact_id = contact_id.split(",")
        mask_all = mask_all.detach().cpu().numpy()
        contact_id = np.array(contact_id)
        mask_all = np.array(mask_all)
        contact_id = contact_id[mask_all]
        
        loss   = criterion(output, label)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        label = label.detach().cpu().numpy()
        pred = nn.Sigmoid()(output).detach().cpu().numpy()
        
        for cid, p, l in zip(contact_id, pred, label):
            preds[cid].append(p)
            labels[cid] = l
            
    label_all = []
    pred_all = []
    for cid, pred in preds.items():
        p = np.mean(pred)
        pred_all.append(p)
        label_all.append(labels[cid])
    #print(labels.shape, preds.shape)
    label_all = np.array(label_all)
    pred_all = np.array(pred_all)
    
    label_all = (label_all > 0.5).astype(int)
    if show:
        best_proba, mcc, y_pred = eval_mcc(label_all, pred_all, show=show)
        print("best_proba", best_proba)
    else:
        mcc = eval_mcc(label_all, pred_all, show=show)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, mcc

# %%
def prepare_loaders(fold, debug=False):
    train_data = []
    valid_data = []
    for i in range(5):
        if i == fold:
            valid_data += train_list[i]
        else:
            train_data += train_list[i]

    train_dataset = NFLDataset(train_data, train_aug, 'train')
    valid_dataset = NFLDataset(valid_data, valid_aug, 'test')

    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], shuffle=True, num_workers=CFG['num_workers'])#, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'])#, pin_memory=True)
    
    return train_loader, valid_loader
# %%
if DEBUG:
    CFG["n_fold"] = 1
    CFG['epochs'] = 1
    for fold in range(5):
        train_list[fold] = train_list[fold][:100]

# %%
best_loss_list = []
best_mcc_list = []

for fold in range(CFG["n_fold"]):
    
    print(f'num_worker = {CFG["num_workers"]}')
    
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)
    
    best_loss = np.inf
    best_mcc = -np.inf
    
    seed_everything(CFG['seed'])
    
    with wandb.init(project='NFL-contact', group=f'{exp_id.split(".")[0]}', name=f'fold{fold}') as run:

        train_loader, valid_loader = prepare_loaders(fold)

        model = NFLModel(backbone = CFG["backbone"],n_segment=CFG["num_channels"]).to(device)

        optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(train_loader)*CFG["epochs"], eta_min=1e-6)

        for epoch in range(1, CFG["epochs"] + 1):
            print()
            print(f"Epoch {epoch}")
            train_one_epoch(model, optimizer, scheduler, dataloader=train_loader, epoch=epoch)
            
        best_loss_list.append(best_loss)
        best_mcc_list.append(best_mcc)

best_loss = np.array(best_loss_list)
best_mcc = np.array(best_mcc_list)

print(best_loss_list)
print(best_mcc_list)
print(f"mcc : {np.mean(best_mcc)}")
print(f"loss : {np.mean(best_loss)}")

# %%
with open(input_path + f"train_{CFG['num_channels']}-{CFG['step_pred']}_dist2_std_sg5folds.pkl", "rb") as tf:
    train_list = pickle.load(tf)

if DEBUG:
    for fold in range(5):
        train_list[fold] = train_list[fold][:100]

# %%
@torch.no_grad()
def feature_extract(model, train_loader, valid_loader):
    
    preds = {}
    train_preds = defaultdict(list)
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for step, (img1,img2, feature, label, mask, contact_id, padding_mask) in pbar:
        img1 = img1.to(device, dtype=torch.float)
        img2 = img2.to(device, dtype=torch.float)
        feature  = feature.to(device, dtype=torch.float)
        mask  = mask.to(device, dtype=torch.bool)
        mask = mask.view(-1)
        padding_mask  = padding_mask.to(device, dtype=torch.bool)
        padding_mask = padding_mask.view(-1)
        
        output = model(img1, img2, feature).view(-1)
        # 評価はdistance < 2のみで行う
        mask_all = mask | padding_mask
        output = output[mask_all]
        
        # batch * num_channels
        contact_id = ",".join(contact_id[0])
        contact_id = contact_id.split(",")
        mask_all = mask_all.detach().cpu().numpy()
        contact_id = np.array(contact_id)
        mask_all = np.array(mask_all)
        contact_id = contact_id[mask_all]
        
        pred = output.detach().cpu().numpy()
        
        for cid, p in zip(contact_id, pred):
            train_preds[cid].append(p)
    
    for cid, pred in train_preds.items():
        p = np.mean(pred)
        preds[cid] = p
        
    train_df = pd.DataFrame({"contact_id":preds.keys(), "logits":preds.values()})
    train_df["is_train"] = 1
    
    preds = {}
    valid_preds = defaultdict(list)
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Test ')
    for step, (img1,img2, feature, label, mask, contact_id, padding_mask) in pbar:
        img1 = img1.to(device, dtype=torch.float)
        img2 = img2.to(device, dtype=torch.float)
        feature  = feature.to(device, dtype=torch.float)
        mask  = mask.to(device, dtype=torch.bool)
        mask = mask.view(-1)
        padding_mask  = padding_mask.to(device, dtype=torch.bool)
        padding_mask = padding_mask.view(-1)
        
        output = model(img1, img2, feature).view(-1)
        # 評価はdistance < 2のみで行う
        mask_all = mask | padding_mask
        output = output[mask_all]
        
        # batch * num_channels
        contact_id = ",".join(contact_id[0])
        contact_id = contact_id.split(",")
        mask_all = mask_all.detach().cpu().numpy()
        contact_id = np.array(contact_id)
        mask_all = np.array(mask_all)
        contact_id = contact_id[mask_all]
        
        pred = output.detach().cpu().numpy()
        
        for cid, p in zip(contact_id, pred):
            valid_preds[cid].append(p)

    for cid, pred in valid_preds.items():
        p = np.mean(pred)
        preds[cid] = p
        
    valid_df = pd.DataFrame({"contact_id":preds.keys(), "logits":preds.values()})
    valid_df["is_train"] = 0
    
    df = pd.concat([train_df, valid_df])
    
    df.to_csv(os.path.join(save_dir, f"cnn_logits_{fold}.csv"))
    print("save cnn logits")
    
    return train_df, valid_df


# %%
for fold in range(CFG["n_fold"]):
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)
    
    train_loader, valid_loader = prepare_loaders(fold)

    model = NFLModel(backbone = CFG["backbone"],n_segment=CFG["num_channels"]).to(device)
    
    _model_path = os.path.join(save_dir, f"best_mcc-{fold:02d}.bin")
    print(_model_path)
    
    model.load_state_dict(torch.load(_model_path))
    model.eval()
    
    feature_extract(model, train_loader, valid_loader)
    
    # debug
    # break

# %%



