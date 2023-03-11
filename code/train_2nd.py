import argparse
import ast
import gc
import glob
import math
import os
import pickle
import random
import sys
import warnings
from pathlib import Path
from time import time
from typing import List

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import yaml
from scipy.optimize import minimize
from sklearn import metrics, model_selection
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path" , type=str, required=True)
    parser.add_argument("--debug" , type=bool, required=False,default=False)
    return parser.parse_args()

args = parse_args()
DEBUG = args.debug
with open(args.config_path, 'r') as f:
    CFG = yaml.safe_load(f)
exp_id = args.config_path.split("/")[-1].split(".")[0]

DATA_DIR = Path("./../data/")
ENSAMBLE_DIR = Path("./../model/")
output_dir = "./../model/" + exp_id.split(".")[0]
OUTPUT_DIR = Path(output_dir)

os.makedirs(output_dir, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def expand_contact_id(df):
    game_play_cols = ["game", "play"]
    df = (
        df.with_columns(pl.col("game_play").str.split("_").alias("game_play_"))
        .with_columns(
            pl.struct(
                [
                    pl.col("game_play_").arr.get(i).alias(game_play_cols[i])
                    for i in range(2)
                ]
            )
            .cast(int)
            .alias("game_play_")
        )
        .unnest("game_play_")
    )
    return df


def add_feature_cols(df_, FEATURE_COLS, remove_col_list):
    additional_cols = list(df_.columns)
    additional_cols = [col for col in additional_cols if not col in remove_col_list]
    FEATURE_COLS.extend(additional_cols)
    return FEATURE_COLS


def target_merge_tracking(
    target_df,
    tracking_df,
    FEATURE_COLS,
    TRACKING_COLS=[
        "game_play",
        "nfl_player_id",
        "step",
        "x_position",
        "y_position",
        "datetime",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
        "team",
        "jersey_number",
        "position",
    ],
):
    target_df = target_df.with_columns(
        pl.concat_str(
            [
                pl.col("game_play"),
                pl.col("step").cast(str),
                pl.col("nfl_player_id_1"),
            ],
            separator="_",
        ).alias("game_step_player_1")
    )
    target_df = target_df.with_columns(
        pl.concat_str(
            [
                pl.col("game_play"),
                pl.col("step").cast(str),
                pl.col("nfl_player_id_2"),
            ],
            separator="_",
        ).alias("game_step_player_2")
    )

    tracking_df = tracking_df.select(TRACKING_COLS)
    tracking_df = tracking_df.with_columns(
        pl.concat_str(
            [
                pl.col("game_play"),
                pl.col("step").cast(str),
                pl.col("nfl_player_id"),
            ],
            separator="_",
        ).alias("game_step_player")
    )

    tracking_df = tracking_df.drop(["game_play", "step", "nfl_player_id", "datetime"])

    # merge tracking to target
    for player_id in [1, 2]:
        tracking_player = tracking_df.select([pl.all().suffix(f"_{player_id}")])
        target_df = target_df.join(
            tracking_player, on=[f"game_step_player_{player_id}"], how="left"
        )
        # add features col
        FEATURE_COLS = add_feature_cols(
            tracking_player,
            FEATURE_COLS,
            [
                f"game_step_player_{player_id}",
                f"frame_{player_id}",
                f"datetime_{player_id}",
            ],
        )
    # drop col
    target_df = target_df.drop(["game_step_player_1", "game_step_player_2"])
    print(len(target_df.columns))
    print("original length", len(target_df))

    # distance
    new_column = (
        np.sqrt(
            (pl.col("x_position_1") - pl.col("x_position_2")) ** 2
            + (pl.col("y_position_1") - pl.col("y_position_2")) ** 2
        )
    ).alias("distance")
    target_df = target_df.with_columns(new_column)
    
    new_column = ((pl.col("step") / 10 * 59.94 + 5 * 59.94).cast(int) + 1).alias("frame")
    target_df = target_df.with_columns(new_column)
    return target_df, FEATURE_COLS

class AbstractBaseBlock:
    def fit_transform(self, input_df: pl.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError()

CLUSTERS = [10, 50, 100, 250, 500]
# CLUSTERS = [10]


def add_step_pct(df, cluster):
    df = df.with_columns(
        np.ceil(
            cluster
            * (pl.col("step") - pl.col("step").min())
            / (pl.col("step").max() - pl.col("step").min())
        )
        .cast(int)
        .alias("step_pct")
    )
    return df


class HelmetBlock(AbstractBaseBlock):
    def __init__(self, baseline_helmets=''):
        self.baseline_helmets = baseline_helmets

    def fit_transform(self, input_df, y=None):

        return self.transform(input_df)

    def transform(self, input_df):
        df = input_df.clone()
        col_len = len(df.columns)

        for cluster in CLUSTERS:
            df = df.groupby("game_play", maintain_order=True).apply(
                lambda x: add_step_pct(x, cluster)
            )
            for helmet_view in ["Sideline", "Endzone"]:
                helmet_train = self.baseline_helmets
                helmet_train = helmet_train.rename({"frame": "step"})
                helmet_train = helmet_train.groupby(
                    "game_play", maintain_order=True
                ).apply(lambda x: add_step_pct(x, cluster))
                helmet_train = helmet_train.filter(pl.col("view") == helmet_view)
                helmet_train = helmet_train.with_columns(
                    (
                        pl.col("game_play")
                        + "_"
                        + pl.col("nfl_player_id").cast(str)
                        + "_"
                        + pl.col("step_pct").cast(str)
                    ).alias("helmet_id")
                )

                helmet_train = (
                    helmet_train.select(
                        pl.col(["helmet_id", "left", "width", "top", "height"])
                    )
                    .groupby("helmet_id", maintain_order=True)
                    .mean()
                )

                for player_ind in [1, 2]:

                    df = df.with_columns(
                        (
                            pl.col("game_play")
                            + "_"
                            + pl.col("nfl_player_id_" + str(player_ind)).cast(str)
                            + "_"
                            + pl.col("step_pct").cast(str)
                        ).alias("helmet_id")
                    )

                    df = df.join(helmet_train, how="left", on="helmet_id")

                    df = df.rename(
                        {
                            i: i
                            + "_"
                            + helmet_view
                            + "_"
                            + str(cluster)
                            + "_"
                            + str(player_ind)
                            for i in ["left", "width", "top", "height"]
                        },
                    )

        return df.select(df.columns[col_len + 2 :])

class RawFeatureBlock(AbstractBaseBlock):
    def __init__(self):
        pass

    def fit_transform(self, input_df, y=None):

        return self.transform(input_df)

    def transform(self, input_df):

        columns = [
            "frame",
            "contact",
            "jersey_number_1",
            "jersey_number_2",
            "distance",
            "cnn_pred",
        ]
        sensor_cols = [
            "x_position",
            "y_position",
            "speed",
            "distance",
            "direction",
            "orientation",
            "acceleration",
            "sa",
            # "coordinate",
        ]
        sensor_cols1 = [s + "_1" for s in sensor_cols]
        sensor_cols2 = [s + "_2" for s in sensor_cols]
        sensor_cols_diff = [f"frame_std_{s}" for s in sensor_cols]

        columns = columns + sensor_cols1 + sensor_cols2

        return input_df.select(columns)

class SimpleCalcBlock(AbstractBaseBlock):
    def __init__(self):
        pass

    def fit_transform(self, input_df, y=None):

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = input_df.clone()
        col_len = len(out_df.columns)

        cols = [
            "x_position",
            "y_position",
            "speed",
            "distance",
            "direction",
            "orientation",
            "acceleration",
            "sa",
            # "coordinate",
        ]

        for i in cols:
            # # diff
            out_df = out_df.with_columns(
                np.abs(pl.col(i + "_1") - pl.col(i + "_2")).alias(i + "_diff")
            )
            # prod
            out_df = out_df.with_columns(
                (pl.col(i + "_1") * pl.col(i + "_2")).alias(i + "_prod")
            )

            # meandiff これ全体のmeanと差を出す意味あるのかな
            out_df = out_df.with_columns(
                (pl.col(i + "_1") * pl.col(i + "_1").mean()).alias(f"diff_ave_{i}_1")
            )
            out_df = out_df.with_columns(
                (pl.col(i + "_2") * pl.col(i + "_2").mean()).alias(f"diff_ave_{i}_2")
            )

            # meandiff
            out_df = out_df.with_columns(
                (pl.col(i + "_1") - pl.col(f"frame_mean_{i}")).alias(
                    f"frame_mean_diff_{i}_1"
                )
            )
            out_df = out_df.with_columns(
                (pl.col(i + "_2") - pl.col(f"frame_mean_{i}")).alias(
                    f"frame_mean_diff_{i}_2"
                )
            )

        # めんどくさいので一旦パス
        # game_rank = pd.DataFrame()
        # game_rank["game_play"] = df["game_play"].unique()
        # game_rank["rank"] = (
        #     df[["game", "play", "datetime"]][~df["game_play"].duplicated()]
        #     .groupby(["game"], as_index=False)["datetime"]
        #     .rank()
        # ).values
        # out_df["rank"] = pd.merge(df, game_rank, how="left", on="game_play")["rank"]
        return out_df.select(out_df.columns[col_len:])

class AggCalcBlock(AbstractBaseBlock):
    def __init__(self):
        pass

    def fit_transform(self, input_df, y=None):

        return self.transform(input_df)

    def transform(self, input_df):
        df = input_df.clone()
        gby_cols = [
            "position_1",
            "position_2",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "team_1",
            "team_2",
            "step",
        ]
        tgt_cols = [
            "distance",
            "cnn_pred",
            "sa_1",
            "distance_1",
            "speed_1",
        ]
        out_dfs = []
        for gby_col in gby_cols:
            for col in tgt_cols:

                agg_df = (
                    df.select(pl.col(["game_play", gby_col, col]))
                    .groupby(["game_play", gby_col], maintain_order=True)
                    .agg(
                        [
                            pl.mean(col).suffix(
                                "_" + "_".join(["game_play", gby_col]) + "_mean"
                            ),
                            pl.min(col).suffix(
                                "_" + "_".join(["game_play", gby_col]) + "_min"
                            ),
                            pl.max(col).suffix(
                                "_" + "_".join(["game_play", gby_col]) + "_max"
                            ),
                            pl.std(col).suffix(
                                "_" + "_".join(["game_play", gby_col]) + "_std"
                            ),
                        ]
                    )
                )

                cols = agg_df.columns[2:]

                out_df = df.join(agg_df, on=["game_play", gby_col], how="left").select(
                    cols
                )
                out_dfs.append(out_df)

        out_df = pl.concat(out_dfs, how="horizontal")

        cols = [s for s in out_df.columns if "_max" in s]
        for col in cols:

            out_df = out_df.with_columns(
                (pl.col(col) - pl.col(col.replace("_max", "_min"))).alias(
                    col.replace("_max", "_range")
                )
            )

        return out_df

class SeqAggCalcBlock(AbstractBaseBlock):
    def __init__(self):
        pass

    def fit_transform(self, input_df, y=None):

        return self.transform(input_df)

    def transform(self, input_df):
        df = input_df.clone()

        df = df.with_columns(pl.col("nfl_player_id_2").fill_null(-1))

        tgt_cols = [
            "distance",
            "cnn_pred",
            "sa_1",
            "distance_1",
            "speed_1",
        ]
        out_dfs = []
        for col in tgt_cols:

            agg_df = (
                df.select(
                    pl.col(["game_play", "nfl_player_id_1", "nfl_player_id_2", col])
                )
                .groupby(
                    ["game_play", "nfl_player_id_1", "nfl_player_id_2"],
                    maintain_order=True,
                )
                .agg(
                    [
                        pl.mean(col).suffix(
                            "_" + "_".join(["game_play", "id_1_2", col]) + "_mean"
                        ),
                        pl.min(col).suffix(
                            "_" + "_".join(["game_play", "id_1_2", col]) + "_min"
                        ),
                        pl.max(col).suffix(
                            "_" + "_".join(["game_play", "id_1_2", col]) + "_max"
                        ),
                        pl.std(col).suffix(
                            "_" + "_".join(["game_play", "id_1_2", col]) + "_std"
                        ),
                    ]
                )
            )
            cols = agg_df.columns[3:]

            out_df = df.join(
                agg_df,
                on=["game_play", "nfl_player_id_1", "nfl_player_id_2"],
                how="left",
            ).select(cols)
            out_dfs.append(out_df)

        out_df = pl.concat(out_dfs, how="horizontal")

        cols = [s for s in out_df.columns if "_max" in s]
        for col in cols:

            out_df = out_df.with_columns(
                (pl.col(col) - pl.col(col.replace("_max", "_min"))).alias(
                    col.replace("_max", "_range")
                )
            )

        out_dfs = []
        # df = df.select(
        #     pl.col(df.columns).sort_by(
        #         ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        #     )
        # )
        for n in [-3, -2, -1, 1, 2, 3]:
            for col in tgt_cols:

                df = df.with_columns(
                    pl.col(col)
                    .shift(n)
                    .over(["game_play", "nfl_player_id_1", "nfl_player_id_2"])
                    .alias(col + f"_shift_{n}")
                )

                out_dfs.append(df.select(col + f"_shift_{n}"))

                df = df.with_columns(
                    (pl.col(col) - pl.col(col + f"_shift_{n}")).alias(
                        col + f"_diff_{n}"
                    )
                )

                out_dfs.append(df.select(col + f"_diff_{n}"))

        out_df = pl.concat([out_df] + out_dfs, how="horizontal")
        return out_df


class LabelEncodingBlock(AbstractBaseBlock):
    def __init__(self):
        self.encoding_dict = {}

        self.columns = [
            "position",
            "nfl_player_id",
            # "team",
        ]

    def fit_transform(self, input_df, y=None):
        for col in self.columns:

            self.encoding_dict[col] = {
                c: i
                for i, c in enumerate(
                    sorted(
                        list(
                            set(
                                input_df.select(
                                    pl.col(f"{col}_1")
                                    .fill_null("nan")
                                    .cast(str)
                                    .unique()
                                )
                                .to_numpy()
                                .reshape(-1)
                            )
                            | set(
                                input_df.select(
                                    pl.col(f"{col}_2")
                                    .fill_null("nan")
                                    .cast(str)
                                    .unique()
                                )
                                .to_numpy()
                                .reshape(-1)
                            )
                        )
                    )
                )
            }

            self.encoding_dict[col] = pl.DataFrame(
                {
                    col: list(self.encoding_dict[col].keys()),
                    f"LE_{col}": list(self.encoding_dict[col].values()),
                }
            )
        return self.transform(input_df)

    def transform(self, input_df):
        out_dfs = []
        df = input_df.clone()

        for col in self.columns:
            df = df.with_columns(pl.col(f"{col}_1").cast(str))
            df = df.join(
                self.encoding_dict[col], how="left", left_on=f"{col}_1", right_on=col
            ).rename({f"LE_{col}": f"LE_{col}_1"})
            out_dfs.append(df.select(f"LE_{col}_1"))

            df = df.with_columns(pl.col(f"{col}_2").cast(str))
            df = df.join(
                self.encoding_dict[col], how="left", left_on=f"{col}_2", right_on=col
            ).rename({f"LE_{col}": f"LE_{col}_2"})
            out_dfs.append(df.select(f"LE_{col}_2"))

        out_df = pl.concat(out_dfs, how="horizontal")

        df = df.with_columns((pl.col("team_1") == pl.col("team_2")).alias("same_team"))
        out_df = pl.concat([out_df, df.select("same_team")], how="horizontal")

        return out_df

def to_feature(input_df, feature_blocks, y=None, istrain=False):
    """input_df を特徴量行列に変換した新しいデータフレームを返す."""

    # processors = [get_num_features, get_category_features, get_simplecalc_features]

    out_df = pl.DataFrame()

    for block in tqdm(feature_blocks, total=len(feature_blocks)):
        with Timer(prefix="" + str(block) + " "):
            # _df = func(input_df, dataType)

            if istrain:
                _df = block.fit_transform(input_df, y=y)
            else:
                _df = block.transform(input_df)

        # 長さが等しいことをチェック (ずれている場合, func の実装がおかしい)
        assert len(_df) == len(input_df), str(block)
        out_df = pl.concat([out_df, _df], how="horizontal")
    #     out_df = utils.reduce_mem_usage(out_df)

    return out_df


def merge_tracking_describe(train, train_player_tracking):

    use_cols = [
        "game_play",
        "step",
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    tracking_describe = (
        train_player_tracking.groupby(["game_play", "step"], maintain_order=True)
        .agg(pl.col(use_cols[2:]).mean())
        .select([pl.col(use_cols[:2]), pl.col(use_cols[2:]).prefix("frame_mean_")])
    )
    train = train.join(tracking_describe, on=["game_play", "step"], how="left")

    tracking_describe = (
        train_player_tracking.groupby(["game_play", "step"], maintain_order=True)
        .agg(pl.col(use_cols[2:]).std())
        .select([pl.col(use_cols[:2]), pl.col(use_cols[2:]).prefix("frame_std_")])
    )
    train = train.join(tracking_describe, on=["game_play", "step"], how="left")
    
    return train

def mainseq_2nd(fold):
    
    print(f'fold = {fold}')
    
    set_seed(777)

    target_dtypes = {
        "contact_id": str,
        "game_play": str,
        "datetime": str,
        "step": int,
        "nfl_player_id_1": str,
        "nfl_player_id_2": str,
        "contact": int,
    }
    FEATURE_COLS = ["nfl_player_id_1", "nfl_player_id_2", "step"]

    ## read data
    train_player_tracking = pl.read_csv(DATA_DIR / "train_player_tracking.csv")
    train_labels = pl.read_csv(DATA_DIR / "train_labels.csv", dtypes=target_dtypes)

    train_baseline_helmets = pl.read_csv(DATA_DIR / "train_baseline_helmets.csv")
    Endzone2_videos = [
        s for s in train_baseline_helmets.get_column("video").unique() if "Endzone2" in s
    ]
    train_baseline_helmets = train_baseline_helmets.filter(
        ~train_baseline_helmets.get_column("video").is_in(Endzone2_videos)
    )

    train_video_metadata = pl.read_csv(DATA_DIR / "train_video_metadata.csv")
    
    ## preprocess
    train_labels = expand_contact_id(train_labels)
    train, feature_cols = target_merge_tracking(
        train_labels, train_player_tracking, FEATURE_COLS
    )
    train = merge_tracking_describe(train, train_player_tracking)

    train = pl.concat(
        [train.filter(pl.col("distance") <= 2), train.filter(pl.col("distance").is_null())]
    )
    
    ## read cnn logits and add to train
    path = CFG['cnn_pred_dir']
    path = ENSAMBLE_DIR / f'{path}/cnn_logits_{fold}.csv'
    cnn_pred_df = pl.read_csv(
        path,
        dtypes=target_dtypes,
    )
    
    train = train.join(
        cnn_pred_df.with_columns(pl.col("logits").alias("cnn_pred")).select(
            pl.col(["contact_id", "cnn_pred", "is_train"])
        ),
        on="contact_id",
        how="left",
    )
    
    trn_df = train.filter(pl.col("is_train") == 1)
    val_df = train.filter(pl.col("is_train") == 0)

    trn_df = trn_df.select(
        pl.col(trn_df.columns).sort_by(["game_play", "nfl_player_id_1", "nfl_player_id_2"])
    )
    val_df = val_df.select(
        pl.col(val_df.columns).sort_by(["game_play", "nfl_player_id_1", "nfl_player_id_2"])
    )

    ## add reverse data
    tgt_cols = [s for s in trn_df.columns if "_1" in s]
    col1 = tgt_cols
    col2 = [s.replace("_1", "_2") for s in tgt_cols]

    trn_df_2 = trn_df.clone()
    tmp = trn_df_2.select(pl.col(col1 + col2)).clone()
    trn_df_2 = trn_df_2.with_columns(pl.DataFrame(tmp[col2].to_pandas(), schema=col1))
    trn_df_2 = trn_df_2.with_columns(pl.DataFrame(tmp[col1].to_pandas(), schema=col2))
    add_trn_len = len(trn_df_2)

    val_df_2 = val_df.clone()
    tmp = val_df_2.select(pl.col(col1 + col2)).clone()
    val_df_2 = val_df_2.with_columns(pl.DataFrame(tmp[col2].to_pandas(), schema=col1))
    val_df_2 = val_df_2.with_columns(pl.DataFrame(tmp[col1].to_pandas(), schema=col2))
    add_val_len = len(val_df_2)

    trn_df_2 = trn_df_2.with_columns(
        [pl.col("jersey_number_1").cast(int), pl.col("jersey_number_2").cast(int)]
    )
    val_df_2 = val_df_2.with_columns(
        [pl.col("jersey_number_1").cast(int), pl.col("jersey_number_2").cast(int)]
    )

    trn_df = pl.concat([trn_df, trn_df_2])
    val_df = pl.concat([val_df, val_df_2])
    
    
    ## feature engineering
    feature_blocks = [
        RawFeatureBlock(),
        SimpleCalcBlock(),
        LabelEncodingBlock(),
        AggCalcBlock(),
        HelmetBlock(train_baseline_helmets),
        SeqAggCalcBlock()
    ]
    
    trn_feature = to_feature(trn_df, feature_blocks, istrain=True)
    val_feature = to_feature(val_df, feature_blocks, istrain=False)
    
    ## train
    trn_feature = trn_feature.to_pandas()
    val_feature = val_feature.to_pandas()

    y_train = trn_feature["contact"]
    X_train = trn_feature.drop(columns=["contact"])

    y_val = val_feature["contact"]
    X_val = val_feature.drop(columns=["contact"])
    
    if CFG['lightgbm']:
        
        params_lgbm_binary = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "max_depth": 8,
            "learning_rate": 0.005,
            "subsample": 0.72,
            "subsample_freq": 4,
            "feature_fraction": 0.4,
            "lambda_l1": 1,
            "lambda_l2": 1,
            "verbose": -1,
            "seed": CFG['seed'],
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params_lgbm_binary,
            train_data,
            valid_sets=[train_data, val_data],
            verbose_eval=250,
            # early_stopping_rounds=200,
            num_boost_round=4000,
        )

        preds_train = model.predict(X_train)
        preds_valid = model.predict(X_val)
    
    else:
        model = xgb.XGBClassifier( 
            n_estimators=4000,
            max_depth=3, 
            learning_rate=0.005, 
            subsample=0.8,
            colsample_bytree=0.4, 
            missing=-1, 
            eval_metric='auc',
            # USE CPU
            # nthread=4,
            # USE GPU
            tree_method='gpu_hist',
            random_state=CFG['seed']
        )

        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=250)

        preds_train = model.predict_proba(X_train)[:, 1]
        preds_valid = model.predict_proba(X_val)[:, 1]


    ## calculate score
    preds_valid[:add_val_len] = (preds_valid[:add_val_len] + preds_valid[-add_val_len:]) / 2
    preds_valid = preds_valid[:-add_val_len]
    y_val = y_val[:-add_val_len]
    val_df = val_df[:-add_val_len]
    val_df = val_df.with_columns(pl.Series(preds_valid).alias("pred"))
    
    def func(x_list):
        score = metrics.matthews_corrcoef(
            val_df.get_column("contact"), val_df.get_column("pred") > x_list[0]
        )
        return -score

    x0 = [0.5]
    result = minimize(func, x0, method="nelder-mead")
    th = result.x[0]
    mcc = metrics.matthews_corrcoef(
        val_df.get_column("contact"), val_df.get_column("pred") > result.x[0]
    )
    print(f"small dataset th = {th}, mcc = {mcc}")
    
    train_labels = expand_contact_id(
        pl.read_csv(DATA_DIR / "train_labels.csv", dtypes=target_dtypes)
    )
    uni_game = val_df.get_column("game").unique().to_list()
    val_whole_df = train_labels.filter(pl.col("game").is_in(uni_game))
    val_whole_df = val_whole_df.join(
        val_df.select(["contact_id", "pred"]),
        how="left",
        on=["contact_id"],
    )
    val_whole_df = val_whole_df.with_columns(pl.col("pred").fill_null(0))

    def func(x_list):
        score = metrics.matthews_corrcoef(
            val_whole_df.get_column("contact"), val_whole_df.get_column("pred") > x_list[0]
        )
        return -score

    x0 = [0.5]
    result = minimize(func, x0, method="nelder-mead")
    th = result.x[0]
    mcc = metrics.matthews_corrcoef(
        val_whole_df.get_column("contact"), val_whole_df.get_column("pred") > result.x[0]
    )
    print(f"whole dataset th = {th}, mcc = {mcc}")

    ## save model and prediction
    with open(OUTPUT_DIR / f"feature_blocks{fold}.pickle", mode="wb") as f:
        pickle.dump(feature_blocks, f)

    with open(OUTPUT_DIR / f"model{fold}.pickle", mode="wb") as f:
        pickle.dump(model, f)

    val_df.select(["contact_id", "contact", "pred"]).write_csv(OUTPUT_DIR / f"oof{fold}.csv")

    preds_train[:add_trn_len] = (preds_train[:add_trn_len] + preds_train[-add_trn_len:]) / 2
    preds_train = preds_train[:-add_trn_len]
    trn_df = trn_df[:-add_trn_len]

    trn_df = trn_df.with_columns(pl.Series(preds_train).alias("pred"))
    trn_df.select(["contact_id", "contact", "pred"]).write_csv(OUTPUT_DIR / f"trn{fold}.csv")

    
if __name__ == "__main__":
    
    logits_paths = sorted((ENSAMBLE_DIR / CFG['cnn_pred_dir']).rglob('cnn_logits_*.csv'))
    folds = len(logits_paths)
    for i in range(folds):
        mainseq_2nd(i)