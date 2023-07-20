# imports
import rasterio
import importlib
import geopandas as gpd
import os
from geopandas import GeoDataFrame, GeoSeries
import glob
import cv2
import pandas as pd

from pathlib import Path
from importlib.metadata import version
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import MyTrainer, setup_cfg
from detectree2.models import predict
from detectree2.models import outputs
from detectree2.models import train
importlib.reload(outputs)
importlib.reload(predict)
importlib.reload(train)
from detectree2.models.outputs import project_to_geojson, clean_crowns, stitch_crowns, filename_geoinfo
from detectree2.models.predict import predict_on_data, DefaultPredictor

# define folders
# site_folder = '../data/Bamberg_Hain/'
site_folder = '../data/Bamberg_Stadtwald/'
#site_folder = '../data/Tretzendorf/'
tiles_dir = site_folder + 'inference/inference_tiles_V3/'

out_dir_test = site_folder + 'inference/outputs_V3/' # check before each use!

# img to make predictions
#img_path = site_folder + 'inference/orig_orthos/Hain_Agisoft_Orthophoto_2_wa.tif' # old version
img_path = site_folder + 'inference/orig_orthos/Stadtwald_Juli-22_final_wa.tif'
#img_path = site_folder + 'inference/orig_orthos/stadtwald_cropped_compressed.tif'
#img_path = site_folder + 'inference/orig_orthos/test_stadtwald_cropped_compressed.tif' # test set
#img_path = site_folder + 'inference/orig_orthos/tretzendorf.tif'
data = rasterio.open(img_path)

folders = [site_folder, tiles_dir, out_dir_test]
for f in folders:
    print(os.path.isdir(f))

# define models and weights
#pre_trained_model = '../data/models/220723_withParacouUAV.pth'
pre_trained_model = '../data/models/230103_randresize_full.pth'
# base model
base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" #with api
# own trained model
trained_model_path = '../data/outputs/model_8.pth'

# specify tiling, choose parameters from training
buffer = 30
tile_width = 40
tile_height = 40

# stitch crowns together
predictions_dir = site_folder + 'inference/inference_tiles_V3/predictions_geo/'
crs_code = 25832

files = glob.glob(predictions_dir + '*geojson' )
print(len(files))

# cleaning steps
crowns = stitch_crowns(predictions_dir,1)
crowns = crowns[crowns.is_valid]
crowns = clean_crowns(crowns, 0.6)
crowns = GeoDataFrame(crowns, geometry='geometry', crs=crs_code)
# TODO: implement function that does cleaning steps until nothing removed
# TODO: update stitch crowns and change pd append to concat
crowns = outputs.remove_low_score_crowns(crowns,0.6)
crowns = outputs.remove_very_small_polygons(crowns,1.0)
crowns = outputs.remove_overlapping_crowns(crowns,0.8)
crowns = outputs.remove_low_score_crowns(crowns,0.6)
crowns = outputs.remove_very_small_polygons(crowns,1.0)
crowns = outputs.remove_overlapping_crowns(crowns,0.8)
crowns = outputs.remove_low_score_crowns(crowns,0.6)
crowns = outputs.remove_very_small_polygons(crowns,1.0)
crowns = outputs.remove_overlapping_crowns(crowns,0.8)
crowns = outputs.remove_very_small_polygons(crowns,1.0)
crowns = outputs.remove_overlapping_crowns(crowns,0.8)
crowns = outputs.remove_very_small_polygons(crowns,1.0)

# define output path and name
output_dir = site_folder + 'inference/outputs_V3/'
name = 'stadtwald_complete_crowns_out.gpkg'

# save shapefile
#crowns = GeoDataFrame(crowns, geometry='geometry', crs=crs_code)
crowns = crowns.drop('index', axis=1)
crowns.to_file(output_dir + name, driver='GPKG')
print("File saved at", output_dir)