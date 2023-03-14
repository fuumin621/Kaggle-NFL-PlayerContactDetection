# Kaggle-NFL-PlayerContactDetection

## Hardware
Google Cloud Platform
* Debian 10.12
* a2-highgpu-1g (vCPU x 12, memory 85 GB)
* 1 x NVIDIA Tesla A100

## Data download
Download data to ./data from https://www.kaggle.com/competitions/nfl-player-contact-detection/data and unzip it.

## Environment
docker-compose up -d --build

## Folder structure
```
.
├── Dockerfile
├── README.md
├── code
│   ├── postprocess.py
│   ├── preprocess_1st.py
│   ├── train.sh
│   ├── train_1st.py
│   └── train_2nd.py
├── config
│   ├── NFL_effnet-b0-TSM-end_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug.yaml
│   ├── NFL_effnet-b0-TSM-end_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug_2nd_lgbm.yaml
│   ├── NFL_effnet-b0-TSM-end_LSTM_64-64_cp-3wh_few_128_drop06_flow_aug.yaml
│   ├── NFL_effnet-b0-TSM-end_LSTM_64-64_cp-3wh_few_128_drop06_flow_aug_2nd_lgbm.yaml
│   ├── NFL_postprocess.yaml
│   ├── NFL_predict.yaml
│   ├── NFL_resnext50-TSM_LSTM_16-16_cp-3wh_few_128_drop04_flow_aug.yaml
│   ├── NFL_resnext50-TSM_LSTM_16-16_cp-3wh_few_128_drop04_flow_aug_2nd_xgboost.yaml
│   ├── NFL_resnext50-TSM_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug.yaml
│   ├── NFL_resnext50-TSM_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug_2nd_lgbm.yaml
│   ├── NFL_resnext50-TSM_LSTM_64-64_cp-3wh_few_128_drop04_flow_aug.yaml
│   ├── NFL_resnext50-TSM_LSTM_64-64_cp-3wh_few_128_drop04_flow_aug_2nd_lgbm.yaml
│   └── NFL_resnext50-TSM_LSTM_64-64_cp-3wh_few_128_drop04_flow_aug_2nd_xgboost.yaml
├── data
├── docker-compose.yml
└── model
```

- Place the downloaded data in the ./data directory.
- The config folder already contains the config files necessary to reproduce the best model.
- The model folder contains the trained models and out-of-fold predictions. The file names used in the config will be used as the folder names for the models.

## Train

### Preprocess
- By executing the following command, preprocessed data will be stored in the ./data directory. 
- This needs to be done before conducting the 1st stage of training (it does not need to be executed again for each training).

```
cd ./code
python preprocess_1st.py
```

### Train 1st stage 
- In the config file, you can specify the backbone model and other settings.

```
cd ./code
python train_1st.py --config [1st stage config_path]
```

### Train 2nd stage 
- In the config file, you can specify the model and which CNN predictions to use from that model.

```
cd ./code
python train_2nd.py --config [2nd stage config_path]
```

### Postprocess
- In the post-processing step, the OOF ensemble score and the optimal threshold for the specified model are calculated.
- In the config file, you can specify the model(s) to be used for the ensemble.
- If the list of models is empty, all models will be considered.
- If 'is_select' is true, [the algorithm](https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private/notebook) will automatically select the model(s) with the highest OOF score.


```
cd ./code
python postprocess.py --config [postprocess config_path]
```

### Script to reproduce the best model
```
cd ./code
bash train.sh
```

## Submit
- Upload the following to kaggle datasets
  - all trained models below ./model
  - config/NFL_predict.yaml
  - data/standard_scaler_dist2.pkl
- Note the following when editing the config file
  - The 'thresh' parameter should be set to the percentile obtained in the post-processing step
  - In 'cnn_configs', please specify a model that matches the 'num_channels' used in the 1st stage.

Here is our dataset and config for final sub.

https://www.kaggle.com/datasets/yuyuki11235/nfl2023-model

Run this notebook.

https://www.kaggle.com/code/nomorevotch/nfl2023-final-submit-5th




