from paths import PROJECT_ROOT
from pathlib import Path
from detectree2.preprocessing import tiling
from detectree2.models import train
from PIL import Image

import os
os.environ['USE_PYGEOS'] = '0'


import glob
import random
import rasterio
import geopandas as gpd
import inspect
import cv2
import PIL
import importlib
importlib.reload(tiling)
importlib.reload(train)

from detectree2.preprocessing.tiling import tile_data_train, to_traintest_folders
from detectree2.models.train import register_train_data, MyTrainer, setup_cfg

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog
)

dataset_dicts_train = DatasetCatalog.get('Schiefer_train')

for d in dataset_dicts_train:
    for obj in d['annotations']:
        if 'segmentation' not in obj:
            print(f'{d["file_name"]} has an annotation with no segmentation field')