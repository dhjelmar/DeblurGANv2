#!/c/Users/dlhje/anaconda3/envs/py39/python

#%% 
## import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re


#%%
# Sharpen (unblur) images
# https://www.youtube.com/watch?v=RcXjx60pso0
# https://github.com/amrrs/pixel-unblur-python/blob/main/pixel-like-unblur-image-deblur-with-deblurganv2.ipynb
# git config --global http.postBuffer 157286400
# git clone https://github.com/VITA-Group/DeblurGANv2
# pip install -r ./requirements.txt
# pip install wget
# download blurred-image.jpeg from https://github.com/amrrs/madebygoogle22-transcript/raw/main/blurred-image.jpeg 
testdir = 'F:\\Documents\\01_Dave\\Programs\\GitHub_home\\DeblurGANv2'
os.chdir(testdir)

# print image to screen
Image.open('blurred-image.jpeg')

#%%
# download pretrained model weights file fpn_inception.h5 
# https://drive.usercontent.google.com/download?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&export=download&authuser=0

# installs required for predict.py
# conda install opencv

# certificate expired for a required download so did this manually
#    http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth
# error message in code looked for it in following spot so put a copy there
# C:/Users/dlhje/.cache/torch/hub/checkpoints/inceptionresnetv2-520b38e4.pth

# needed torch to be installed with CUDA
# https://pytorch.org/
# wants to install the following, but that is a problem because I do not have nvidia
#    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# instead, tried
#    conda install pytorch torchvision torchaudio cpuonly -c pytorch
# cpuonly does not work; likely fails with model.cuda() and img.cuda() calls
# Possibly try following to use my AMD GPU; would probably need to repace above cuda calls
#    https://www.xda-developers.com/nvidia-cuda-amd-zluda/
# Probably better to try on Google Colab


# test the code
# added following to 1st line of python.py script so uses specified environment
# #!/c/Users/dlhje/anaconda3/envs/py39/python
# run outside IDE as
#    ./predict.py blurry-image.jpg
# or here as follows
! ./predict.py blurry-image.jpg


# %%
