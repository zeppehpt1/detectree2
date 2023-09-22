# used to train bakim detectree2 model
import os
os.environ['USE_PYGEOS'] = '0'

from detectree2.models.train import register_train_data, MyTrainer, setup_cfg

# register training data
train_location = "../data/bakim/Bamberg_Hain/tiles_40_30_0.5" + "/train/"
#train_location = "../data/bakim/Bamberg_Hain/tiles_99_0_0.2" + "/train/"
assert os.path.isdir(train_location)
register_train_data(train_location, 'Bamberg_Hain', val_fold=5) # will take on all folds because not enough data

train_location = "../data/bakim/Bamberg_Stadtwald/tiles_40_30_0.5" + "/train/"
#train_location = "../data/bakim/Bamberg_Stadtwald/tiles_99_0_0.2" + "/train/"
assert os.path.isdir(train_location)
register_train_data(train_location, "Bamberg_Stadtwald", val_fold=5)

train_location = "../data/bakim/Tretzendorf_upper/tiles_40_30_0.5" + "/train/"
#train_location = "../data/bakim/Tretzendorf_upper/tiles_99_0_0" + "/train/"
assert os.path.isdir(train_location)
register_train_data(train_location, "Tretzendorf_upper", val_fold=5)

train_location = "../data/bakim/Tretzendorf_lower/tiles_40_30_0.5" + "/train/"
#train_location = "../data/bakim/Tretzendorf_lower/tiles_99_0_0" + "/train/"
assert os.path.isdir(train_location)
register_train_data(train_location, "Tretzendorf_lower", val_fold=5)

# supply base model from detectron model zoo
# set base model
base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" #with api
#pre_trained_model = site_folder + 'models/220723_withParacouUAV.pth'
pre_trained_model = '../data/thesis/models/230103_randresize_full.pth'

# supply registered sets
# registered sets
trains = ("Bamberg_Hain_train", "Bamberg_Stadtwald_train", "Tretzendorf_upper_train", "Tretzendorf_lower_train")
tests = ("Bamberg_Hain_val", "Bamberg_Stadtwald_val", "Tretzendorf_upper_val", "Tretzendorf_lower_val")

out_dir = "../data/bakim/outputs/model3"

cfg = setup_cfg(base_model,
                trains,
                tests, workers=4,
                eval_period=50,
                update_model=str(pre_trained_model),
                max_iter=3000,
                out_dir=str(out_dir),
                resize=True) # update_model arg can be used to load in trained  model

trainer = MyTrainer(cfg, patience=6)
trainer.resume_or_load(resume=False)
trainer.train()