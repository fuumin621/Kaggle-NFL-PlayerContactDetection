import os
import gc
import yaml
import argparse
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
from tqdm import tqdm

from sklearn.metrics import matthews_corrcoef
from numba import jit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path" , type=str, required=True)
    parser.add_argument("--debug" , type=bool, required=False,default=False)
    return parser.parse_args()

args = parse_args()
DEBUG = args.debug
with open(args.config_path, 'r') as f:
    CFG = yaml.safe_load(f)
model_ids = CFG['model_ids']

@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


# def eval_mcc(y_true, y_prob, show=False):
@jit
def eval_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    best_percentile = 0
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
                best_percentile = i / n
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    return best_mcc, best_proba, best_percentile

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_total_pred(paths, iscnn=True, ispred=True):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        if iscnn:
            df = df[(df["is_train"] == 0)&(df["contact_id"]!="padding_id")]
            df["pred"]= df["logits"].apply(lambda x:sigmoid(x))
        dfs.append(df)
    dfs = pd.concat(dfs,ignore_index=True).sort_values('contact_id')
    if ispred:
        return dfs["pred"].values
    else:
        return dfs["contact_id"].values
    
def make_total_pred(model_dir, ispred=True):
    cnn_pred_paths = sorted(glob(model_dir + 'cnn_logits*.csv'))
    if len(cnn_pred_paths) > 0:
        pred_paths = cnn_pred_paths
        iscnn = True
    tree_pred_paths = sorted(glob(model_dir + 'oof*.csv'))
    if len(tree_pred_paths) > 0:
        pred_paths = tree_pred_paths
        iscnn = False
    pred = read_total_pred(pred_paths, iscnn, ispred)
    return pred

model_path = "./../model/**/"
input_path = "./../data/"

if __name__ == '__main__':

    labels = pd.read_csv(input_path + "train_labels.csv")[["contact_id", "contact"]]
    labels = pd.Series(labels["contact"].values, index=labels["contact_id"].values)
    labels = labels.to_dict()

    if len(model_ids) == 0:
        model_dirs = sorted(glob(model_path))
        model_ids = [p.split("/")[-2] for p in model_dirs]
    else:
        model_dirs = [f"./../model/{id_}/" for id_ in model_ids]
    print(model_ids)

    oof_df = pd.DataFrame()
    for model_dir, model_id in tqdm(zip(model_dirs, model_ids)):
        pred = make_total_pred(model_dir)
        oof_df[model_id] = pred
    oof_df['contact_id'] = make_total_pred(model_dir, ispred=False)
    oof_df['contact'] = oof_df['contact_id'].map(labels)

    # 単体スコア計算
    target = oof_df["contact"].values
    best_id = None
    best_score = -1
    for exp_id in model_ids:
        score, _, _ = eval_mcc(target, oof_df[exp_id].values)
        print("{}, score:{:.4f}".format(exp_id, score))
        if score > best_score:
            best_id = exp_id
            best_score = score

    print("single best, {}, {:.4f}".format(best_id,best_score))

    l = 0
    best_ids = [best_id]
    while True:
        print(f"start loop {l}")
        best_id = None
        best_score, _, _ = eval_mcc(target, oof_df[best_ids].mean(axis=1).values)
        for exp_id in model_ids:
            if exp_id not in best_ids:
                score, _, _ = eval_mcc(
                    target, oof_df[best_ids + [exp_id]].mean(axis=1).values
                )
                print("{}, score:{:.4f}".format(exp_id, score))
                if score > best_score:
                    best_id = exp_id
                    best_score = score
        if best_id is not None:
            best_ids.append(best_id)
            print("best_id:{}, best_score:{:.4f}".format(best_id, best_score))
            print("")
            l += 1
        else:
            print("stop")
            break

    oof_df["pred"] = oof_df[best_ids].mean(axis=1)
    score, thresh, percent = eval_mcc(target, oof_df["pred"].values)

    print("best_ids:{}".format(best_ids))
    print("best_score:{:.4f}".format(score))
    print("best_thresh:{:.4f}".format(thresh))
    print("best_percent:{:.4f}".format(percent))