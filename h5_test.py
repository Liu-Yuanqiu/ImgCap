import os
import json
import redis
import base64, struct, numpy
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
h5path = os.path.join(path, cap + "swin_dert_grid_train.h5")
with h5py.File(h5path, 'w') as hf:
    hf.create_dataset('label', (L, len(text_field.vocab)), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid', (L, 60, 1024), dtype=numpy.float32)
    hf.create_dataset('swin_dert_grid_mask', (L, 1, 1, 60), dtype=numpy.bool_)
    # hf.create_dataset('up_down_36', (L, 36, 2048))
    hf.create_dataset('token_kd', (L, 20), dtype=numpy.int32)
    hf.create_dataset('cap_gt', (L, 5), dtype=h5py.special_dtype(vlen=str))
with h5py.File("../mscoco/swin_dert_grid_train.h5", 'r') as hf:
    labels = hf['label']
    swin_dert_grids = hf['swin_dert_grid']
    swin_dert_grid_masks = hf['swin_dert_grid_mask']
    token_kds = hf['token_kd']
    cap_gts = hf['cap_gt']
    print(labels.shape)
    print(labels[1000].sum())
    print(swin_dert_grids[10000].sum())
    print(swin_dert_grid_masks[10000])
    print(cap_gts[20000])
    print(token_kds[20000])