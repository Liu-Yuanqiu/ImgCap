import os
import json
import h5py
import numpy
from data.field import TextField
from data.utils import load_txt
path = '../mscoco/'
cap = 'swin_dert_grid'
for img_id_path in ["coco_test4w_image_id.txt", "coco_val3w_image_id.txt"]:
    img_ids = load_txt(os.path.join(path, "txt", img_id_path))
    h5path = os.path.join(path, cap + "_" + img_id_path.split('_')[1] +".h5")
    print(h5path)
    L = len(img_ids)
    with h5py.File(h5path, 'w') as hf:
        hf.create_dataset('image_id', data=img_ids)
        hf.create_dataset('swin_dert_grid', (L, 60, 1024), dtype=numpy.float32)
        hf.create_dataset('swin_dert_grid_mask', (L, 1, 1, 60), dtype=numpy.bool_)

    with h5py.File(h5path, 'a') as hf:
        swin_dert_grids = hf['swin_dert_grid']
        swin_dert_grid_masks = hf['swin_dert_grid_mask']

        for inx, img_id in enumerate(img_ids):
            swin_dert_grid_path = os.path.join(path, "feature", "swin_dert_grid", str(img_id) + ".npz")
            with numpy.load(swin_dert_grid_path, allow_pickle=True) as data:
                grid_sd = data['grid']
                grid_sd = numpy.array(grid_sd).astype('float32')
                grid_sd_mask = data['mask']
                grid_sd_mask = numpy.array(grid_sd_mask).astype('bool')
                swin_dert_grids[inx] = grid_sd
                swin_dert_grid_masks[inx] = grid_sd_mask
                # img_ids[inx] = numpy.int32(img_id)
            
            if inx==1000:
                print(img_ids[1000])
                print(swin_dert_grids[1000].sum())
                print(swin_dert_grid_masks[1000])