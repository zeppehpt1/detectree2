from paths import PROJECT_ROOT
from pathlib import Path
from detectree2.preprocessing import tiling
from detectree2.models import train

import detectree2.preprocessing.tiling
import detectree2.models.train

import importlib
importlib.reload(tiling)
importlib.reload(train)

from detectree2.preprocessing.tiling import tile_data_train, to_traintest_folders
from detectree2.models.train import register_train_data, MyTrainer, setup_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog
)

print("ALL registered datasets")
print("\n")
baba = DatasetCatalog.register
print(baba)