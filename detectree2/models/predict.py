"""Generate predictions.

This module contains the code to generate predictions on tiled data.
"""
import json
import os
from pathlib import Path

import cv2
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from tqdm import tqdm

from detectree2.models.train import get_filenames, get_tree_dicts

# Code to convert RLE data from the output instances into Polygons,
# a small amout of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here


def predict_on_data(
    directory: str = "./",
    out_folder: str = "predictions",
    predictor=DefaultPredictor,
    eval=False,
    save: bool = True,
    num_predictions=0,
):
    """Make predictions on tiled data.

    Predicts crowns for all png images present in a directory and outputs masks as jsons.
    """

    pred_dir = os.path.join(directory, out_folder)

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if eval:
        dataset_dicts = get_tree_dicts(directory)
    else:
        dataset_dicts = get_filenames(directory)

    total_files = len(dataset_dicts)

    # Works out if all items in folder should be predicted on
    if num_predictions == 0:
        num_to_pred = len(dataset_dicts)
    else:
        num_to_pred = num_predictions

    print(f"Predicting {num_to_pred} files")

    for i, d in enumerate(dataset_dicts[:num_to_pred], start=1):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        # Creating the file name of the output file
        file_name_path = d["file_name"]
        # Strips off all slashes so just final file name left
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_name = file_name.replace("png", "json")
        output_file = os.path.join(pred_dir, f"Prediction_{file_name}")
        # print(output_file)

        if save:
            # Converting the predictions to json files and saving them in the
            # specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)

        if i % 50 == 0:
            print(f"Predicted {i} files of {total_files}")


if __name__ == "__main__":
    predict_on_data()
