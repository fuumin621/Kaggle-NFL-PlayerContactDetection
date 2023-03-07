# %%
import os
import sys
import glob
import numpy as np
import pandas as pd
import random
import pickle
import math
import gc
import cv2
from tqdm import tqdm
import time
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from multiprocessing import Pool


# %%
input_path = "./../data/"
frame_path = "./../data/frames/train/"


# %%
def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

labels = pd.read_csv(input_path + "train_labels.csv")
tracking = pd.read_csv(input_path + "train_player_tracking.csv")
helmets = pd.read_csv(input_path + "train_baseline_helmets.csv")
video_metadata = pd.read_csv(input_path + "train_video_metadata.csv")


#%%
# ### split frames
def split_frames(video):
    if 'Endzone2' not in video:
        os.system(f"ffmpeg -i ./../data/train/{video} -q:v 2 -f image2 {frame_path}{video}_%04d.jpg -hide_banner -loglevel error")
os.makedirs(frame_path,exist_ok=True)
videos = helmets.video.unique()
num_workers = 12
run = split_frames
args = videos
with Pool(processes=num_workers) as pool:
    imap = pool.imap_unordered(run,args)
    dfs = list(tqdm(imap, total=len(args)))

# %%
# ### video2helmets
video2helmets = {}
helmets_new = helmets.set_index('video')
for video in tqdm(helmets.video.unique()):
    video2helmets[video] = helmets_new.loc[video].reset_index(drop=True)
with open(input_path + "video2helmets.pkl", "wb") as tf:
    pickle.dump(video2helmets,tf)

#%%
# ### video2frames
video2frames = {}
for game_play in tqdm(video_metadata.game_play.unique()):
    for view in ['Endzone', 'Sideline']:
        video = game_play + f'_{view}.mp4'
        video2frames[video] = max(list(map(lambda x:int(x.split('_')[-1].split('.')[0]), \
                                           glob.glob(f'{frame_path}{video}*'))))
with open(input_path + "video2frames.pkl", "wb") as tf:
    pickle.dump(video2frames,tf)

### make std dist2
# %%
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
        
#     df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
#     output_cols += ["G_flug"]
    return df_combo, output_cols


#%%

use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa', 'team'
]

skip_cols = ["team"]
df, feature_cols = create_features(labels, tracking, use_cols=use_cols)
train = df.copy()

# %%
train["distance"] = train["distance"].fillna(-1)
train_filtered = train.query('not distance>2').reset_index(drop=True)

# %%
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
scaler.fit(train_filtered[scale_cols])
# %%
import pickle
with open(input_path + "standard_scaler_dist2.pkl", "wb") as tf:
    scaler = pickle.dump(scaler, tf)


# %%
# make datasets
df['G_flug'] = (df['nfl_player_id_2']=="G")
df["same_team"] = (df['team_1'] == df['team_2']) & (~df['G_flug'])
df["different_team"] = (df['team_1'] != df['team_2']) & (~df['G_flug'])
df["same_team"] = df["same_team"].astype(int)
df["different_team"] = df["different_team"].astype(int)

# %%
def contact_combination(x):
    x = x.split("_")
    x = "_".join([x[0], x[1], x[3], x[4]])
    return x
df["contact_unique"] = df["contact_id"].apply(contact_combination)
df["game"] = df["game_play"].apply(lambda x: x.split("_")[0])
df["frame"] = (df['step']/10*59.94+5*59.94).astype('int')+1
df["distance_below_2"] = df["distance"] < 2 
df["distance_below_1.5"] = df["distance"] < 1.5

skf = StratifiedGroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["contact"], groups = df["game"])):
    df.loc[val_idx, 'fold'] = fold

# %%
df.groupby(['fold','contact'])['game_play'].count()

# %%
with open(input_path + "standard_scaler_dist2.pkl", "rb") as tf:
    scaler = pickle.load(tf)
df[scale_cols] = scaler.transform(df[scale_cols])

# %%
contact_unique = {contact: group.reset_index(drop=True) for contact, group in df.groupby('contact_unique')}

# %%
with open(input_path + "contact_unique_dist2_std_sg5folds.pkl", "wb") as tf:
    pickle.dump(contact_unique,tf)

# %%
def get_dummy_row(df, frame):
    x = frame - len(df)
    empty_df = pd.DataFrame([df.iloc[-1].values] * x, columns=df.columns)
    empty_df["contact_id"] = "padding_id"
    df = pd.concat([df, empty_df], ignore_index=True).interpolate(limit_direction='both')
    return df
def make_datasets(ch,step):
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
        # print(fold)

        len_set = set()
        pos = 0
        neg = 0
        for df in train_fold_data[fold]:
            len_set.add(len(df))
            pos += df["contact"].sum()
            neg += len(df) - df["contact"].sum()
        #     if len(df) != 64:
        #         print(df)
        #         break
    with open(input_path + f"train_{ch}-{step}_dist2_std_sg5folds.pkl", "wb") as tf:
        pickle.dump(train_fold_data,tf)

# %%
make_datasets(ch=32,step=32)
make_datasets(ch=32,step=8)
make_datasets(ch=64,step=64)
make_datasets(ch=16,step=16)
make_datasets(ch=16,step=4)

# %%


# %%



