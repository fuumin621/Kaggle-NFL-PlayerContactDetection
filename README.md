# Kaggle-NFL-PlayerContactDetection
## NN part

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
bash train_1st.sh
```