from paths import PROJECT_ROOT
from pathlib import Path
from detectree2.preprocessing import tiling
from detectree2.models import train

import torch
import importlib
import os
os.environ['USE_PYGEOS'] = '0'
importlib.reload(tiling)
importlib.reload(train)

from detectree2.models.train import register_train_data, MyTrainer, setup_cfg
from detectron2.data import (
    DatasetCatalog
)

# check if gpu is used
if torch.cuda.current_device() == 1 or 2:
    print("all used, exit training now")
    exit
print("gpu free to use")

# folder setup
# site_folder = PROJECT_ROOT / 'data' / 'Bamberg_Hain' laptop
site_folder = '/home/nieding/data/Bamberg_Hain/'
site_name = 'Schiefer'
out_dir = site_folder + 'outputs'
Path(out_dir).mkdir(parents=True, exist_ok=True)

# # remove dataset before creation --> debugging purposes
# dataset_name = site_name + '_train'
# if dataset_name in DatasetCatalog.list():
#     DatasetCatalog.remove(dataset_name)
# dataset_name = site_name + '_val'
# if dataset_name in DatasetCatalog.list():
#     DatasetCatalog.remove(dataset_name)
# print("Datasets removed")

# register datasets
train_location = site_folder + 'tiles/train/'
register_train_data(train_location, site_name,1) # registers train and val sets
print("Datasets registered")

# supply base model from detectron model zoo
# set base model
base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" #with api
pre_trained_model = site_folder + 'models/220723_withParacouUAV.pth'

# train model
trains = ("Schiefer_train",)
tests = ("Schiefer_val",)

cfg = setup_cfg(base_model,trains, tests, workers=4, eval_period=100, update_model=str(pre_trained_model), max_iter=3000, out_dir=str(out_dir)) # update_model arg can be used to load in trained  model
trainer = MyTrainer(cfg, patience=4)
trainer.resume_or_load(resume=False)
trainer.train()