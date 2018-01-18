import numpy as np, os
from collections import OrderedDict

CONFIG = {}
CONFIG["data"] = {}
CONFIG["log"] = {}
CONFIG["pretrained"] = {}  # should refactor when using another model architecture
CONFIG["training"] = {}
CONFIG["seed"] = 42

# Logging
CONFIG["log"]["root"] = "log"
CONFIG["log"]["log_step"] = 50
CONFIG["log"]["ckpt_step"] = 500
CONFIG["log"]["ckpt_dir"] = "asset/ckpt"

# Data
CONFIG["data"]["root"] = "asset/data"
CONFIG["data"]["npy_path"] = os.path.join(CONFIG["data"]["root"], "data.npy")

# Model
CONFIG["pretrained"]["MEAN_PIXEL"] = np.array([123.68 , 116.779, 103.939])
CONFIG["pretrained"]["weights"] = "asset/densenet121_weights_tf.h5" #denses
CONFIG["pretrained"]["layers"] = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4', 'pool5',

    'fc6', 'relu_6', 'fc7', 'relu_7',
    # 'fc8'
)
CONFIG["pretrained"]["trainable"] = OrderedDict([
    ('conv1_1', False), ('conv1_2', False),
    ('conv2_1', False), ('conv2_2', False),
    ('conv3_1', False), ('conv3_2', False), ('conv3_3', False), ('conv3_4', False),
    ('conv4_1', False), ('conv4_2', False), ('conv4_3', False), ('conv4_4', False),
    ('conv5_1', True), ('conv5_2', True), ('conv5_3', True), ('conv5_4', True),
    ('fc6', True), ('fc7', True),
    # ('fc8', True)
])  #TODO review: NaN loss (Nesterov)!! Also after few training epochs, dev_set accuracy stuck at 0.556 (Adam)!!


# Training
CONFIG["training"]["init_param"] = "imagenet"
CONFIG["training"]["n_epochs"] = 100
CONFIG["training"]["bsize"] = 16  # 1 epoch ~ 1500 steps for this bsize
CONFIG["training"]["init_lr"] = [1e-7]
CONFIG["training"]["l2"] = 5e-4
CONFIG["training"]["p"] = .3
CONFIG["training"]["evaluate_step"] = 3000
CONFIG["training"]["patience"] = 30*CONFIG["training"]["evaluate_step"]
CONFIG["training"]["restore_ckpt_dir"] = ""  # if not empty string, restore from given ckpt_dir
