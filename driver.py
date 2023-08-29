# used to train thesis model
from pathlib import Path
from detectree2.preprocessing import tiling
from detectree2.models import train

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 206049024

import os
os.environ['USE_PYGEOS'] = '0'

import importlib
importlib.reload(tiling)
importlib.reload(train)

from detectree2.models.train import register_train_data, MyTrainer, setup_cfg

# folder setup
site_folder = '../data/'
out_dir = site_folder + 'outputs/test_model'
Path(out_dir).mkdir(parents=True, exist_ok=True)

# # register hain
train_location = '/home/nieding/data/Bamberg_Hain/training/training_tiles/train/'
register_train_data(train_location, 'Bamberg_Hain', 1) # registers train and val sets
print("Hain datasets registered")

# register stadtwald
train_location = '/home/nieding/data/Bamberg_Stadtwald/training/training_tiles/train/'
register_train_data(train_location, 'Bamberg_Stadtwald', 1) # registers train and val sets
print("Stadtwald datasets registered")

# register tretzendorf
train_location = '/home/nieding/data/Tretzendorf/training/training_tiles/train/'
register_train_data(train_location, 'Tretzendorf', 1) # registers train and val sets
print("Tretzendorf datasets registered")

# register schiefer
train_location = '/home/nieding/data/Schiefer/training/training_tiles/train/'
register_train_data(train_location, 'Schiefer', 1) # registers train and val sets
print("Schiefer datasets registered")

# supply base model from detectron model zoo
# set base model
base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" #with api
#pre_trained_model = site_folder + 'models/220723_withParacouUAV.pth'
pre_trained_model = site_folder + 'models/230103_randresize_full.pth'

# registered sets
trains = ("Bamberg_Hain_train", "Bamberg_Stadtwald_train", "Tretzendorf_train", "Schiefer_train")
tests = ("Bamberg_Hain_val", "Bamberg_Stadtwald_val", "Tretzendorf_val", "Schiefer_val")

cfg = setup_cfg(base_model,
                trains,
                tests, workers=4,
                eval_period=20,
                update_model=str(pre_trained_model),
                max_iter=3000,
                out_dir=str(out_dir),
                resize=True) # update_model arg can be used to load in trained  model

trainer = MyTrainer(cfg, patience=10)
trainer.resume_or_load(resume=False)
trainer.train()