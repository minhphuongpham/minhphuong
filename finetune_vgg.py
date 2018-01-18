from __future__ import division, print_function
import numpy as np, tensorflow as tf, os

from configs import CONFIG
from data.preprocess import DataLoader
from model.vgg19 import VGG19
from utils import init_project, report_assessment

from collections import OrderedDict

init_project()  # setup path to data/assets/.., create some tmp dir, etc

### Load dataset
dataLoader = DataLoader(data_configs=CONFIG["data"])
# dataLoader.prepare_data(debug_n_images=20)  # run once to prepare data in the right format, and save to disk
train_set, dev_set, test_set = dataLoader.load_data()

### Modelling and Training
# Setup options
# init_opts
finetuned_layers = [layer for layer, v in CONFIG["pretrained"]["trainable"].iteritems() if v is True]
vgg19_init_opts = OrderedDict([("p", CONFIG["training"]["p"]), ])

# Init model
model = VGG19(
    img_shape=(224, 224, 3),
    n_classes=dataLoader.n_classes,
    init_param="imagenet",
    finetuned_layers=finetuned_layers,
    opts=vgg19_init_opts,
    loss="hinge", init_lr=CONFIG["training"]["init_lr"], l2=CONFIG["training"]["l2"]
)
# Training
model.fit(train_set, dev_set, **CONFIG["training"])

### Assessment
print("\nTEST SET PERFORMANCE")
model.evaluate(test_set, bsize=CONFIG["training"]["bsize"])
print("\nFinal dev set performance")
model.evaluate(dev_set, bsize=CONFIG["training"]["bsize"])
print("\nFinal train set performance")
model.evaluate(train_set, bsize=CONFIG["training"]["bsize"])

report_assessment(model.meta)

model.close()

#TODO script the hyperparam search & log hyperparam settings + assement results of each setting
