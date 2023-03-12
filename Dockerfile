FROM gcr.io/kaggle-gpu-images/python:latest
# for A100
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1  -f https://download.pytorch.org/whl/torch_stable.html
RUN sudo apt-get update -y && sudo apt-get install -y libturbojpeg
RUN pip install timm==0.6.12
RUN pip install PyTurboJPEG==1.7.0
RUN pip install polars==0.16.8
