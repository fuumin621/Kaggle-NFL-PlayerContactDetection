import gc
import glob
import math
import os
import pickle
import random
import sys
import time
from functools import lru_cache
from multiprocessing import Pool

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


input_path = "./../data/"
frame_path = "./../data/frames/train/"
os.makedirs(frame_path,exist_ok=True)

use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa', 'team'
]
skip_cols = ["team"]

scale_cols = [
'x_position_1',
 'y_position_1',
 'speed_1',
 'distance_1',
 'direction_1',
 'orientation_1',
 'acceleration_1',
 'sa_1',
 'x_position_2',
 'y_position_2',
 'speed_2',
 'distance_2',
 'direction_2',
 'orientation_2',
 'acceleration_2',
 'sa_2',
 'distance',
]
channels_and_steps = [
    (64,64),
    (32,32),
    (32,8),
    (16,16),
    (16,4),
]

def read_data():
    labels = pd.read_csv(input_path + "train_labels.csv")
    tracking = pd.read_csv(input_path + "train_player_tracking.csv")
    helmets = pd.read_csv(input_path + "train_baseline_helmets.csv")
    video_metadata = pd.read_csv(input_path + "train_video_metadata.csv")
    return labels,tracking,helmets,video_metadata

def split_frames(video):
    if 'Endzone2' not in video:
        os.system(f"ffmpeg -i ./../data/train/{video} -q:v 2 -f image2 {frame_path}{video}_%04d.jpg -hide_banner -loglevel error")

def run_multi_process(args,func,num_workers):
    with Pool(processes=num_workers) as pool:
        imap = pool.imap_unordered(func,args)
        dfs = list(tqdm(imap, total=len(args)))

def create_video2helmets(helmets):
    video2helmets = {}
    helmets_new = helmets.set_index('video')
    for video in tqdm(helmets.video.unique()):
        video2helmets[video] = helmets_new.loc[video].reset_index(drop=True)
    with open(input_path + "video2helmets.pkl", "wb") as tf:
        pickle.dump(video2helmets,tf)

def create_video2frames(video_metadat):
    video2frames = {}
    for game_play in tqdm(video_metadata.game_play.unique()):
        for view in ['Endzone', 'Sideline']:
            video = game_play + f'_{view}.mp4'
            video2frames[video] = max(list(map(lambda x:int(x.split('_')[-1].split('.')[0]), \
                                            glob.glob(f'{frame_path}{video}*'))))
    with open(input_path + "video2frames.pkl", "wb") as tf:
        pickle.dump(video2frames,tf)


def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols if c not in skip_cols]
    output_cols += [c+"_2" for c in use_cols if c not in skip_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
    return df_combo

def create_scaler(df):
    df = df.copy()
    df["distance"] = df["distance"].fillna(-1)
    df_filtered = df.query('not distance>2').reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(df_filtered[scale_cols])
    with open(input_path + "standard_scaler_dist2.pkl", "wb") as tf:
        scaler = pickle.dump(scaler, tf)

def contact_combination(x):
    x = x.split("_")
    x = "_".join([x[0], x[1], x[3], x[4]])
    return x

def preprocess_features(df):
    df['G_flug'] = (df['nfl_player_id_2']=="G")
    df["same_team"] = (df['team_1'] == df['team_2']) & (~df['G_flug'])
    df["different_team"] = (df['team_1'] != df['team_2']) & (~df['G_flug'])
    df["same_team"] = df["same_team"].astype(int)
    df["different_team"] = df["different_team"].astype(int)

    df["contact_unique"] = df["contact_id"].apply(contact_combination)
    df["game"] = df["game_play"].apply(lambda x: x.split("_")[0])
    df["frame"] = (df['step']/10*59.94+5*59.94).astype('int')+1
    df["distance_below_2"] = df["distance"] < 2 
    df["distance_below_1.5"] = df["distance"] < 1.5

    skf = StratifiedGroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["contact"], groups = df["game"])):
        df.loc[val_idx, 'fold'] = fold
    with open(input_path + "standard_scaler_dist2.pkl", "rb") as tf:
        scaler = pickle.load(tf)
    df[scale_cols] = scaler.transform(df[scale_cols])
    contact_unique = {contact: group.reset_index(drop=True) for contact, group in df.groupby('contact_unique')}
    return contact_unique

def get_dummy_row(df, frame):
    x = frame - len(df)
    empty_df = pd.DataFrame([df.iloc[-1].values] * x, columns=df.columns)
    empty_df["contact_id"] = "padding_id"
    df = pd.concat([df, empty_df], ignore_index=True).interpolate(limit_direction='both')
    return df

def make_datasets(contact_unique,ch,step):
    dummy_num = 0

    train_fold_data = [[] for _ in range(5)]

    for k, v in tqdm(contact_unique.items()):
        fold = int(v.iloc[0]["fold"])
        
        for i in range(0, len(v), step):
            tmp = v.iloc[i:i+ch]
            
            if len(tmp) != ch:
                now = len(v) - ch
                
                if now < 0:
                    tmp = get_dummy_row(tmp, ch)
                    dummy_num += 1
                else:
                    tmp = v.iloc[now:now+ch]
                if tmp.distance_below_2.sum() == 0 and k[-1] != "G":
                    break
                train_fold_data[fold].append(tmp)
                break
                
            if tmp.distance_below_2.sum() == 0 and k[-1] != "G":
                continue
                
            train_fold_data[fold].append(tmp)

    for fold in range(5):
        len_set = set()
        pos = 0
        neg = 0
        for df in train_fold_data[fold]:
            len_set.add(len(df))
            pos += df["contact"].sum()
            neg += len(df) - df["contact"].sum()
    with open(input_path + f"train_{ch}-{step}_dist2_std_sg5folds.pkl", "wb") as tf:
        pickle.dump(train_fold_data,tf)


if __name__ == '__main__':
    labels,tracking,helmets,video_metadata = read_data()
    
    run_multi_process(args=helmets.video.unique(),func=split_frames,num_workers=12)
    create_video2helmets(helmets)
    create_video2frames(video_metadata)
    
    df = create_features(labels, tracking, use_cols=use_cols)
    create_scaler(df)
    contact_unique = preprocess_features(df)
    
    for ch,step in channels_and_steps:
        make_datasets(contact_unique,ch=ch,step=step)







