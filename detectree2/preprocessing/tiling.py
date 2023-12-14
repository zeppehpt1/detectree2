"""Tiling orthomosaic and crown data.

These functions tile orthomosaics and crown data for training and evaluation
of models and making landscape predictions.
"""

import json
import os
import random
import shutil
import warnings
from math import ceil
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from fiona.crs import from_epsg  # noqa: F401
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.mask import mask
from shapely.geometry import box

# class img_data(DatasetReader):
#    """
#    Class for image data to be processed for tiling
#    """
#
#    def __init__(self):
#        self.x_origin = self.bounds[0]
#        self.y_origin = self.bounds[3]
#        self.pixelSizeX = self.affine[0]
#        self.pixelSizeY = -self.affine[4]
#


def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them.

    Args:
      gdf: Input geopandas dataframe

    Returns:
      json style data
    """
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def tile_data(
    data: DatasetReader,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    dtype_bool: bool = False,
) -> None:
    """Tiles up orthomosaic for making predictions on.

    Tiles up full othomosaic into managable chunks to make predictions on. Use tile_data_train to generate tiled
    training data. A bug exists on some input raster types whereby outputed tiles are completely black - the dtype_bool
    argument should be switched if this is the case.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        dtype_bool: Flag to edit dtype to prevent black tiles

    Returns:
        None
    """
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)
    crs = CRS.from_string(data.crs.wkt)
    crs = crs.to_epsg()
    tilename = Path(data.name).stem

    total_tiles = int(
        ((data.bounds[2] - data.bounds[0]) / tile_width) * ((data.bounds[3] - data.bounds[1]) / tile_height)
    )

    tile_count = 0
    print(f"Tiling to {total_tiles} total tiles")

    for minx in np.arange(data.bounds[0], data.bounds[2] - tile_width,
                          tile_width, int):
        for miny in np.arange(data.bounds[1], data.bounds[3] - tile_height,
                              tile_height, int):

            tile_count += 1
            # Naming conventions
            out_path_root = out_path / f"{tilename}_{minx}_{miny}_{tile_width}_{buffer}_{crs}"
            # new tiling bbox including the buffer
            bbox = box(
                minx - buffer,
                miny - buffer,
                minx + tile_width + buffer,
                miny + tile_height + buffer,
            )
            # define the bounding box of the tile, excluding the buffer
            # (hence selecting just the central part of the tile)
            # bbox_central = box(minx, miny, minx + tile_width, miny + tile_height)

            # turn the bounding boxes into geopandas DataFrames
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
            # geo_central = gpd.GeoDataFrame(
            #    {"geometry": bbox_central}, index=[0], crs=from_epsg(4326)
            # )  # 3182
            # overlapping_crowns = sjoin(crowns, geo_central, how="inner")

            # here we are cropping the tiff to the bounding box of the tile we want
            coords = get_features(geo)
            # print("Coords:", coords)

            # define the tile as a mask of the whole tiff with just the bounding box
            out_img, out_transform = mask(data, shapes=coords, crop=True)

            # Discard scenes with many out-of-range pixels
            out_sumbands = np.sum(out_img, 0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.where(out_sumbands == 765, 1, 0)
            sumzero = zero_mask.sum()
            sumnan = nan_mask.sum()
            totalpix = out_img.shape[1] * out_img.shape[2]
            if sumzero > 0.25 * totalpix:
                continue
            elif sumnan > 0.25 * totalpix:
                continue

            out_meta = data.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "nodata": None,
            })
            # dtype needs to be unchanged for some data and set to uint8 for others
            if dtype_bool:
                out_meta.update({"dtype": "uint8"})
            # print("Out Meta:",out_meta)

            # Saving the tile as a new tiff, named by the origin of the tile.
            # If tile appears blank in folder can show the image here and may
            # need to fix RGB data or the dtype
            # show(out_img)
            out_tif = out_path_root.with_suffix(out_path_root.suffix + ".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            # read in the tile we have just saved
            clipped = rasterio.open(out_tif)
            # read it as an array
            # show(clipped)
            arr = clipped.read()

            # each band of the tiled tiff is a colour!
            r = arr[0]
            g = arr[1]
            b = arr[2]

            # stack up the bands in an order appropriate for saving with cv2,
            # then rescale to the correct 0-255 range for cv2

            rgb = np.dstack((b, g, r))  # BGR for cv2

            if np.max(g) > 255:
                rgb_rescaled = 255 * rgb / 65535
            else:
                rgb_rescaled = rgb  # scale to image
            # print("rgb rescaled", rgb_rescaled)

            # save this as jpg or png...we are going for png...again, named with the origin of the specific tile
            # here as a naughty method
            cv2.imwrite(
                str(out_path_root.with_suffix(out_path_root.suffix + ".png").resolve()),
                rgb_rescaled,
            )
            if tile_count % 50 == 0:
                print(f"Processed {tile_count} tiles of {total_tiles} tiles")

    print("Tiling complete")


def tile_data_train(  # noqa: C901
    data: DatasetReader,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    crowns: gpd.GeoDataFrame = None,
    threshold: float = 0,
    nan_threshold: float = 0.1,
    dtype_bool: bool = False,
) -> None:
    """Tiles up orthomosaic and corresponding crowns into training tiles.

    A threshold can be used to ensure a good coverage of crowns across a tile. Tiles that do not have sufficient
    coverage are rejected.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        crowns: Crown polygons as a geopandas dataframe
        threshold: Min proportion of the tile covered by crowns to be accepted {0,1}
        nan_theshold: Max proportion of tile covered by nans
        dtype_bool: Flag to edit dtype to prevent black tiles

    Returns:
        None

    """

    # TODO: Clip data to crowns straight away to speed things up
    # TODO: Tighten up epsg handling
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)
    tilename = Path(data.name).stem
    crs = CRS.from_string(data.crs.wkt)
    crs = crs.to_epsg()
    # out_img, out_transform = mask(data, shapes=crowns.buffer(buffer), crop=True)
    # Should start from data.bounds[0] + buffer, data.bounds[1] + buffer to avoid later complications
    for minx in np.arange(ceil(data.bounds[0]) + buffer, data.bounds[2] - tile_width - buffer, tile_width, int):
        for miny in np.arange(ceil(data.bounds[1]) + buffer, data.bounds[3] - tile_height - buffer, tile_height, int):

            out_path_root = out_path / f"{tilename}_{minx}_{miny}_{tile_width}_{buffer}_{crs}"

            # Calculate the buffered tile dimensions
            # tile_width_buffered = tile_width + 2 * buffer
            # tile_height_buffered = tile_height + 2 * buffer

            # Calculate the bounding box coordinates with buffer
            minx_buffered = minx - buffer
            miny_buffered = miny - buffer
            maxx_buffered = minx + tile_width + buffer
            maxy_buffered = miny + tile_height + buffer

            # Create the affine transformation matrix for the tile
            # transform = from_bounds(minx_buffered, miny_buffered, maxx_buffered,
            #                        maxy_buffered, tile_width_buffered, tile_height_buffered)

            bbox = box(minx_buffered, miny_buffered, maxx_buffered, maxy_buffered)
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
            coords = get_features(geo)

            # Skip if insufficient coverage of crowns - good to have early on to save on unnecessary processing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Warning:
                # _crs_mismatch_warn
                overlapping_crowns = gpd.clip(crowns, geo)

                # Ignore tiles with no crowns
                if overlapping_crowns.empty:
                    continue

                # Discard tiles that do not have a sufficient coverage of training crowns
                if (overlapping_crowns.dissolve().area[0] / geo.area[0]) < threshold:
                    print("discarded a tile")
                    continue

            # define the tile as a mask of the whole tiff with just the bounding box
            out_img, out_transform = mask(data, shapes=coords, crop=True)

            # Discard scenes with many out-of-range pixels
            out_sumbands = np.sum(out_img, 0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.where(out_sumbands == 765, 1, 0)
            sumzero = zero_mask.sum()
            sumnan = nan_mask.sum()
            totalpix = out_img.shape[1] * out_img.shape[2]
            if sumzero > nan_threshold * totalpix:  # reject tiles with many 0 cells
                continue
            elif sumnan > nan_threshold * totalpix:  # reject tiles with many NaN cells
                continue

            out_meta = data.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "nodata": None,
            })
            # dtype needs to be unchanged for some data and set to uint8 for others to deal with black tiles
            if dtype_bool:
                out_meta.update({"dtype": "uint8"})

            # Saving the tile as a new tiff, named by the origin of the tile. If tile appears blank in folder can show
            # the image here and may need to fix RGB data or the dtype
            out_tif = out_path_root.with_suffix(out_path_root.suffix + ".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            # read in the tile we have just saved
            clipped = rasterio.open(out_tif)

            # read it as an array
            arr = clipped.read()

            # each band of the tiled tiff is a colour!
            r = arr[0]
            g = arr[1]
            b = arr[2]

            # stack up the bands in an order appropriate for saving with cv2, then rescale to the correct 0-255 range
            # for cv2. BGR ordering is correct for cv2 (and detectron2)
            rgb = np.dstack((b, g, r))

            # Some rasters need to have values rescaled to 0-255
            # TODO: more robust check
            if np.max(g) > 255:
                rgb_rescaled = 255 * rgb / 65535
            else:
                # scale to image
                rgb_rescaled = rgb

            # save this as png, named with the origin of the specific tile
            # potentially bad practice
            cv2.imwrite(
                str(out_path_root.with_suffix(out_path_root.suffix + ".png").resolve()),
                rgb_rescaled,
            )

            # select the crowns that intersect the non-buffered central
            # section of the tile using the inner join
            # TODO: A better solution would be to clip crowns to tile extent
            # overlapping_crowns = sjoin(crowns, geo_central, how="inner")
            # Maybe left join to keep information of crowns?

            overlapping_crowns = overlapping_crowns.explode(index_parts=True)

            # Translate to 0,0 to overlay on png
            moved = overlapping_crowns.translate(-minx + buffer, -miny + buffer)

            # scale to deal with the resolution
            scalingx = 1 / (data.transform[0])
            scalingy = -1 / (data.transform[4])
            moved_scaled = moved.scale(scalingx, scalingy, origin=(0, 0))

            impath = {"imagePath": out_path_root.with_suffix(out_path_root.suffix + ".png").as_posix()}

            # Save as a geojson, a format compatible with detectron2, again named by the origin of the tile.
            # If the box selected from the image is outside of the mapped region due to the image being on a slant
            # then the shp file will have no info on the crowns and hence will create an empty gpd Dataframe.
            # this causes an error so skip creating geojson. The training code will also ignore png so no problem.
            try:
                filename = out_path_root.with_suffix(out_path_root.suffix + ".geojson")
                moved_scaled = overlapping_crowns.set_geometry(moved_scaled)
                moved_scaled.to_file(
                    driver="GeoJSON",
                    filename=filename,
                )
                with open(filename, "r") as f:
                    shp = json.load(f)
                    shp.update(impath)
                with open(filename, "w") as f:
                    json.dump(shp, f)
            except ValueError:
                print("Cannot write empty DataFrame to file.")
                continue
            # Repeat and want to save crowns before being moved as overlap with lidar data to get the heights
            # can try clean up the code here as lots of reprojecting and resaving but just going to get to
            # work for now
            out_geo_file = out_path_root.parts[-1] + "_geo"
            out_path_geo = out_path / Path(out_geo_file)
            try:
                filename_unmoved = out_path_geo.with_suffix(out_path_geo.suffix + ".geojson")
                overlapping_crowns.to_file(
                    driver="GeoJSON",
                    filename=filename_unmoved,
                )
                with open(filename_unmoved, "r") as f:
                    shp = json.load(f)
                    shp.update(impath)
                with open(filename_unmoved, "w") as f:
                    json.dump(shp, f)
            except ValueError:
                print("Cannot write empty DataFrame to file.")
                continue

    print("Tiling complete")


def image_details(fileroot):
    """Take a filename and split it up to get the coordinates, tile width and the buffer and then output box structure.

    Args:
        fileroot: image filename without file extension

    Returns:
        Box structure
    """
    image_info = fileroot.split("_")
    minx = int(image_info[-5])
    miny = int(image_info[-4])
    tile_width = int(image_info[-3])
    buffer = int(image_info[-2])

    xbox_coords = (minx - buffer, minx + tile_width + buffer)
    ybox_coords = (miny - buffer, miny + tile_width + buffer)
    return [xbox_coords, ybox_coords]


def is_overlapping_box(test_boxes_array, train_box):
    """Check if the train box overlaps with any of the test boxes.

    Args:
        test_boxes_array:
        train_box:

    Returns:
        Boolean
    """
    for test_box in test_boxes_array:
        test_box_x = test_box[0]
        test_box_y = test_box[1]
        train_box_x = train_box[0]
        train_box_y = train_box[1]

        # Check if both the x and y coords overlap meaning the entire box does and hence end loop
        if test_box_x[1] > train_box_x[0] and train_box_x[1] > test_box_x[0]:
            if test_box_y[1] > train_box_y[0] and train_box_y[1] > test_box_y[0]:
                return True

    return False


def record_data(crowns,
                out_dir,
                column='status'):
    """Function that will record a list of classes into a file that can be readed during training.

    Args:
        crowns: gpd dataframe with the crowns
        out_dir: directory to save the file
        column: column name to get the classes from

    Returns:
        None
    """

    list_of_classes = crowns[column].unique().tolist()

    print("**The list of classes are:**")
    print(list_of_classes)
    print("**The list has been saved to the out_dir**")

    # Write it into file "classes.txt"
    out_tif = out_dir + 'classes.txt'
    f = open(out_tif, "w")
    for i in list_of_classes:
        f.write("%s\n" % i)
    f.close()


def to_traintest_folders(  # noqa: C901
        tiles_folder: str = "./",
        out_folder: str = "./data/",
        test_frac: float = 0.2,
        folds: int = 1,
        strict: bool = False,
        seed: int = None) -> None:
    """Send tiles to training (+validation) and test dir

    With "strict" it is possible to automatically ensure no overlap between train/val and test tiles.

    Args:
        tiles_folder: folder with tiles
        out_folder: folder to save train and test folders
        test_frac: fraction of tiles to be used for testing
        folds: number of folds to split the data into
        strict: if True, training/validation files will be removed if there is any overlap with test files (inc buffer)

    Returns:
        None
    """
    tiles_dir = Path(tiles_folder)
    out_dir = Path(out_folder)

    if not os.path.exists(tiles_dir):
        raise IOError

    if Path(out_dir / "train").exists() and Path(out_dir / "train").is_dir():
        shutil.rmtree(Path(out_dir / "train"))
    if Path(out_dir / "test").exists() and Path(out_dir / "test").is_dir():
        shutil.rmtree(Path(out_dir / "test"))
    Path(out_dir / "train").mkdir(parents=True, exist_ok=True)
    Path(out_dir / "test").mkdir(parents=True, exist_ok=True)

    file_names = tiles_dir.glob("*.png")
    file_roots = [item.stem for item in file_names]

    num = list(range(0, len(file_roots)))

    # this affects the random module module-wide
    if seed is not None:
        random.seed(seed)
    random.shuffle(num)

    test_boxes = []

    for i in range(0, len(file_roots)):
        # copy to test
        if i < len(file_roots) * test_frac:
            test_boxes.append(image_details(file_roots[num[i]]))
            shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(
                Path(file_roots[num[i]]).suffix + ".geojson"), out_dir / "test")
        else:
            # copy to train
            train_box = image_details(file_roots[num[i]])
            if strict:   # check if there is overlap with test boxes
                if not is_overlapping_box(test_boxes, train_box):
                    shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(
                        Path(file_roots[num[i]]).suffix + ".geojson"), out_dir / "train")
            else:
                shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(
                    Path(file_roots[num[i]]).suffix + ".geojson"), out_dir / "train")

    # COMMENT NECESSARY HERE
    file_names = (out_dir / "train").glob("*.geojson")
    file_roots = [item.stem for item in file_names]
    # stemname = Path(filenames[0]).stem.split("_", 1)[0]
    # indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]
    # random.shuffle(indices)
    num = list(range(0, len(file_roots)))
    random.shuffle(num)
    ind_split = np.array_split(file_roots, folds)

    for i in range(0, folds):
        Path(out_dir / f"train/fold_{i + 1}").mkdir(parents=True, exist_ok=True)
        for name in ind_split[i]:
            shutil.move(
                out_dir / f"train/{name}.geojson",  # type: ignore
                out_dir / f"train/fold_{i + 1}/{name}.geojson",
            )


if __name__ == "__main__":
    # Right let"s test this first with Sepilok 10cm resolution, then I need to try it with 50cm resolution.
    img_path = "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_benchmark/P4_Ortho_2015.tif"
    crown_path = "gdrive/MyDrive/JamesHirst/NY/Buffalo/Buffalo_raw_data/all_crowns.shp"
    out_dir = "./"
    # Read in the tiff file
    # data = img_data.open(img_path)
    # Read in crowns
    data = rasterio.open(img_path)
    crowns = gpd.read_file(crown_path)
    print(
        "shape =",
        data.shape,
        ",",
        data.bounds,
        "and number of bands =",
        data.count,
        ", crs =",
        data.crs,
    )

    buffer = 20
    tile_width = 200
    tile_height = 200

    tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns)
    to_traintest_folders(folds=5)
