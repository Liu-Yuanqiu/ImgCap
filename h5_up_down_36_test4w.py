import os
import json
import h5py
import numpy
from data.field import TextField
from data.utils import load_txt
path = '../mscoco/'
cap = 'up_down_36'
for img_id_path in ["coco_test4w_image_id.txt", "coco_val3w_image_id.txt"]:
    img_ids = load_txt(os.path.join(path, "txt", img_id_path))
    h5path = os.path.join(path, cap + "_" + img_id_path.split('_')[1] +".h5")
    print(h5path)
    L = len(img_ids)
    with h5py.File(h5path, 'w') as hf:
        hf.create_dataset('image_id', (L,), dtype=numpy.int32)
        hf.create_dataset('up_down_36', (L, 36, 2048), dtype=numpy.float32)

    with h5py.File(h5path, 'a') as hf:
        image_ids = hf['image_id']
        up_down_36s = hf['up_down_36']
        
        for inx, img_id in enumerate(img_ids):
            up_down_36_path = os.path.join(path, "feature", "up_down_36", str(img_id) + ".npz")
            with numpy.load(up_down_36_path, allow_pickle=True) as data:
                region_ud = data['feat']
                region_ud = numpy.array(region_ud).astype('float32')
                up_down_36s[inx] = region_ud
                img_ids[inx] = img_id
            
            if inx==10000:
                print(img_ids[10000])
                print(up_down_36s[10000].sum())