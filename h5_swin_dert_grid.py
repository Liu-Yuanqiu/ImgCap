import os
import json
import h5py
from data.field import TextField
from data.utils import load_txt
def get_stop_words(text_field, stop_word_path):
    words = load_txt(stop_word_path)
    stop_word_ids = []
    for w in words:
        stop_word_ids.append(text_field.vocab.stoi[w])
    return stop_word_ids
path = '../mscoco/'
text_field = TextField(vocab_path=os.path.join(path, "txt", "coco_vocabulary.txt"), vocab_s_path=os.path.join(path, "txt", "se_labels.txt"))
pad_id = text_field.vocab.stoi['<pad>']
MAX_LEN = 20
stop_words = get_stop_words(text_field, os.path.join(path, "txt", "english"))

cap = 'swin_dert_grid'
cached_train = "cached_coco_train_"+cap+".json"
cached_val = "cached_coco_val_"+cap+".json"
cached_test = "cached_coco_test_test_"+cap+".json"

train_samples = json.load(open(os.path.join(path, cached_train), "r"))
val_samples = json.load(open(os.path.join(path, cached_val), "r"))
test_samples = json.load(open(os.path.join(path, cached_test), "r"))

L = len(train_samples)
h5path = os.path.join(path, cap + "_train.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset('label', (L, len(text_field.vocab)), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid', (L, 60, 1024), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid_mask', (L, 1, 1, 60), dtype=numpy.bool_)
    # hf.create_dataset('up_down_36', (L, 36, 2048))
    hf.create_dataset('token_kd', (L, 20), dtype=numpy.int32)
    hf.create_dataset('cap_gt', (L, 5), dtype=h5py.special_dtype(vlen=str))
with h5py.File(h5path, 'a') as hf:
    labels = hf['label']
    swin_dert_grids = hf['swin_dert_grid']
    swin_dert_grid_masks = hf['swin_dert_grid_mask']
    token_kds = hf['token_kd']
    cap_gts = hf['cap_gt']

    for inx, sample in enumerate(train_samples):
        id = sample['id']
        token_gt = sample['token_gt']
        token_kd = sample['token_kd']
        max_len = max( max([len(x) for x in token_gt]), len(token_kd) )
        label = numpy.zeros(len(text_field.vocab), dtype=numpy.float32)
        for i in range(max_len):
            for j in range(len(token_gt)):
                if i >= len(token_gt[j]):
                    pass
                else:
                    wid = token_gt[j][i]
                    if wid not in [0, 1, 2, 3] and wid not in stop_words:
                        label[wid] += 1
                    else:
                        pass
        labels[inx] = label
        cap_gts[inx] = sample['cap_gt'][:5]
        
        token_kd = token_kd[:MAX_LEN]
        token_kd = token_kd + [pad_id] * (MAX_LEN - len(token_kd))
        token_kds[inx] = token_kd

        swin_dert_grid_path = sample['swin_dert_grid_path']
        with numpy.load(swin_dert_grid_path, allow_pickle=True) as data:
            grid_sd = data['grid']
            grid_sd = numpy.array(grid_sd).astype('float32')
            grid_sd_mask = data['mask']
            grid_sd_mask = numpy.array(grid_sd_mask).astype('bool')
            swin_dert_grids[inx] = grid_sd
            swin_dert_grid_masks[inx] = grid_sd_mask
        if inx==10000:
            print(labels[7000].sum())
            print(swin_dert_grids[7000].sum())
            print(swin_dert_grid_masks[7000])
            print(cap_gts[7000])
            print(token_kds[7000])


L = len(val_samples)
h5path = os.path.join(path, cap + "_val.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset('label', (L, len(text_field.vocab)), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid', (L, 60, 1024), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid_mask', (L, 1, 1, 60), dtype=numpy.bool_)
    # hf.create_dataset('up_down_36', (L, 36, 2048))
    hf.create_dataset('token_kd', (L, 20), dtype=numpy.int32)
    hf.create_dataset('cap_gt', (L, 5), dtype=h5py.special_dtype(vlen=str))
with h5py.File(h5path, 'a') as hf:
    labels = hf['label']
    swin_dert_grids = hf['swin_dert_grid']
    swin_dert_grid_masks = hf['swin_dert_grid_mask']
    token_kds = hf['token_kd']
    cap_gts = hf['cap_gt']

    for inx, sample in enumerate(val_samples):
        id = sample['id']
        token_gt = sample['token_gt']
        token_kd = sample['token_kd']
        max_len = max( max([len(x) for x in token_gt]), len(token_kd) )
        label = numpy.zeros(len(text_field.vocab), dtype=numpy.float32)
        for i in range(max_len):
            for j in range(len(token_gt)):
                if i >= len(token_gt[j]):
                    pass
                else:
                    wid = token_gt[j][i]
                    if wid not in [0, 1, 2, 3] and wid not in stop_words:
                        label[wid] += 1
                    else:
                        pass
        labels[inx] = label
        cap_gts[inx] = sample['cap_gt'][:5]
        
        token_kd = token_kd[:MAX_LEN]
        token_kd = token_kd + [pad_id] * (MAX_LEN - len(token_kd))
        token_kds[inx] = token_kd

        swin_dert_grid_path = sample['swin_dert_grid_path']
        with numpy.load(swin_dert_grid_path, allow_pickle=True) as data:
            grid_sd = data['grid']
            grid_sd = numpy.array(grid_sd).astype('float32')
            grid_sd_mask = data['mask']
            grid_sd_mask = numpy.array(grid_sd_mask).astype('bool')
            swin_dert_grids[inx] = grid_sd
            swin_dert_grid_masks[inx] = grid_sd_mask
        if inx==3000:
            print(labels[2000].sum())
            print(swin_dert_grids[2000].sum())
            print(swin_dert_grid_masks[2000])
            print(cap_gts[2000])
            print(token_kds[2000])  

L = len(test_samples)
h5path = os.path.join(path, cap + "_test.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset('label', (L, len(text_field.vocab)), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid', (L, 60, 1024), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid_mask', (L, 1, 1, 60), dtype=numpy.bool_)
    # hf.create_dataset('up_down_36', (L, 36, 2048))
    hf.create_dataset('token_kd', (L, 20), dtype=numpy.int32)
    hf.create_dataset('cap_gt', (L, 5), dtype=h5py.special_dtype(vlen=str))
with h5py.File(h5path, 'a') as hf:
    labels = hf['label']
    swin_dert_grids = hf['swin_dert_grid']
    swin_dert_grid_masks = hf['swin_dert_grid_mask']
    token_kds = hf['token_kd']
    cap_gts = hf['cap_gt']

    for inx, sample in enumerate(test_samples):
        id = sample['id']
        token_gt = sample['token_gt']
        token_kd = sample['token_kd']
        max_len = max( max([len(x) for x in token_gt]), len(token_kd) )
        label = numpy.zeros(len(text_field.vocab), dtype=numpy.float32)
        for i in range(max_len):
            for j in range(len(token_gt)):
                if i >= len(token_gt[j]):
                    pass
                else:
                    wid = token_gt[j][i]
                    if wid not in [0, 1, 2, 3] and wid not in stop_words:
                        label[wid] += 1
                    else:
                        pass
        labels[inx] = label
        cap_gts[inx] = sample['cap_gt'][:5]
        
        token_kd = token_kd[:MAX_LEN]
        token_kd = token_kd + [pad_id] * (MAX_LEN - len(token_kd))
        token_kds[inx] = token_kd

        swin_dert_grid_path = sample['swin_dert_grid_path']
        with numpy.load(swin_dert_grid_path, allow_pickle=True) as data:
            grid_sd = data['grid']
            grid_sd = numpy.array(grid_sd).astype('float32')
            grid_sd_mask = data['mask']
            grid_sd_mask = numpy.array(grid_sd_mask).astype('bool')
            swin_dert_grids[inx] = grid_sd
            swin_dert_grid_masks[inx] = grid_sd_mask
        if inx==3000:
            print(labels[2000].sum())
            print(swin_dert_grids[2000].sum())
            print(swin_dert_grid_masks[2000])
            print(cap_gts[2000])
            print(token_kds[2000])
            
        