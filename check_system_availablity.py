from paths import PROJECT_ROOT
from pathlib import Path
from detectree2.preprocessing import tiling
from detectree2.models import train

import detectree2.preprocessing.tiling
import detectree2.models.train

import os
os.environ['USE_PYGEOS'] = '0'

import torch
import psutil
import importlib
import numpy as np

from detectree2.preprocessing.tiling import tile_data_train, to_traintest_folders
from detectree2.models.train import register_train_data, MyTrainer, setup_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog
)

# conda clean --all to remove unused packages from time to time

# script use to check several things from time to time

# print("ALL registered datasets")
# print("\n")
# baba = DatasetCatalog.register
# print(baba)
print("\n")
print("Is cuda available?",torch.cuda.is_available())
print("How many devices are there?",torch.cuda.device_count())
print("How many devices are currently used",torch.cuda.current_device())
print("First GPUs name",torch.cuda.get_device_name(0))
print("Second GPUs name",torch.cuda.get_device_name(1))
print("First GPU MEM available",torch.cuda.mem_get_info(0))
print("Second GPU MEM available",torch.cuda.mem_get_info(1))
ram = int(np.round(psutil.virtual_memory().total / (1024. **3)))
print("Available RAM",ram, "GB")