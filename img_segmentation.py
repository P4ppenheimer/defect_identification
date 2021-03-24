### load packages
import numpy as np
import random
import os
import warnings
import torch
import segmentation_models_pytorch as smp

# device is cuda if gpu is availabe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")

# set seed and make everything deterministic
seed = 69
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from helper_functions import *
from prediction_function import *

ts = '2021-03-03 11_00_31'

PATH = f'./unet_model/model_{ts}.pth'
unet_model_path = f'./unet_model/model_{ts}.pth'

model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
)

state = torch.load(PATH, map_location = device)
model.load_state_dict(state['state_dict'])
model.to(device)

print(get_prediction(idx=5, model=model))