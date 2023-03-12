# Kaggle-NFL-PlayerContactDetection

### Hardware
Google Cloud Platform
* Debian 10.12
* a2-highgpu-1g (vCPU x 12, memory 85 GB)
* 1 x NVIDIA Tesla A100

### Data download
Download data to ./data from https://www.kaggle.com/competitions/nfl-player-contact-detection/data and unzip it.

### Environment
docker-compose up -d --build

### Preprocess
```
cd ./code
python preprocess_1st.py
```

### train
```
cd ./code
python train_1st.py [train config_path]
```

### postprocess
```
cd ./code
python postprocess.py [postprocess config_path]
```

### script to reproduce the best sub
```
cd ./code
bash train.sh
```
