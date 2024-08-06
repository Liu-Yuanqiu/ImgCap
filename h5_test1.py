import os
import json
import redis
import base64, struct, numpy
import h5py
from data.field import TextField
from data.utils import load_txt
dataset = h5py.File("../mscoco/swin_dert_grid_test4w.h5", 'r')
image_ids = dataset['image_id']
swin_dert_grids = dataset['swin_dert_grid']
print(image_ids[1000])
print(swin_dert_grids[1000].sum())