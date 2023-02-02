from paths import PROJECT_ROOT
from pathlib import Path
from detectree2.preprocessing import tiling
from detectree2.models import train

import importlib
importlib.reload(tiling)
importlib.reload(train)

from detectree2.models.train import register_train_data, MyTrainer, setup_cfg
from detectron2.data import (
    DatasetCatalog
)

site_folder = PROJECT_ROOT / 'data' / 'Bamberg_Hain'

# remove dataset before creation --> debugging purposes
dataset_name = 'Schiefer_train'
if dataset_name in DatasetCatalog.list():
    DatasetCatalog.remove(dataset_name)

dataset_name = 'Schiefer_val'
if dataset_name in DatasetCatalog.list():
    DatasetCatalog.remove(dataset_name)


# register datasets
train_location = str(site_folder / 'tiles' / 'train')
#val_location = str(site_folder / 'tiles' / 'test')
register_train_data(train_location, 'test',1)

# supply base model from detectron model zoo
# set base model
base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" #with api
pre_trained_model = str(site_folder / 'models' / '220723_withParacouUAV.pth')

# trained model
trains = ("Schiefer_train")
tests = ("Schiefer_val")

out_dir = str(site_folder / 'outputs')
Path(out_dir).mkdir(parents=True, exist_ok=True)

# no validation set
cfg = setup_cfg(base_model,tests, trains, workers=4, eval_period=100, update_model=str(pre_trained_model), max_iter=3000, out_dir=str(out_dir)) # update_model arg can be used to load in trained  model

trainer = MyTrainer(cfg, patience=4)
trainer.resume_or_load(resume=False)
trainer.train()