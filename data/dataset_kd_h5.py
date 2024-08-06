import os
import json
import h5py
import numpy as np
import itertools
import collections
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from data.utils import load_txt, nested_tensor_from_tensor_list
from data.transforms import get_transform
from data.field import TextField
from pycocotools.coco import COCO as pyCOCO

class PairedCollator:
    def __init__(self, origin, device='cpu'):
        self.origin = origin
        self.device = device

    def __call__(self, batch):
        # cap_gt, token_kd, label, feat, mask
        outputs = {}
        cap_gt = [item[0] for item in batch]
        outputs['cap_gt'] = cap_gt
        tokens_kd = [item[1].astype(np.int64) for item in batch]
        tokens_kd = torch.from_numpy(np.stack(tokens_kd, 0))
        outputs['token_kd'] = tokens_kd
        labels = [item[2] for item in batch]
        labels = torch.from_numpy(np.stack(labels, 0))
        outputs['label'] = labels

        if self.origin=="swin_dert_grid":
            feat = [item[3] for item in batch]
            feat = torch.from_numpy(np.stack(feat, 0))
            mask = [item[4] for item in batch]
            mask = torch.from_numpy(np.stack(mask, 0))
            outputs['feat'] = feat
            outputs['mask'] = mask
        elif self.origin=="up_down_36":
            feat = [item[3] for item in batch]
            feat = torch.from_numpy(np.stack(feat, 0))
            outputs['feat'] = feat

        return outputs
       
class PairedDataset:
    def __init__(self, path, origin):
        self.path = path
        self.origin = origin
        with h5py.File(path, 'r') as f:
            self.length = len(f['token_kd'])

    def open_h5(self):
        self.dataset = h5py.File(self.path, 'r')
        self.origin = self.origin
        if self.origin=="swin_dert_grid":
            self.feats = self.dataset['swin_dert_grid']
            self.masks = self.dataset['swin_dert_grid_mask']
        elif self.origin=="up_down_36":
            self.feats = self.dataset['up_down_36']
        self.labels = self.dataset['label']
        self.token_kds = self.dataset['token_kd']
        self.cap_gts = self.dataset['cap_gt']

    def __getitem__(self, index):
        if not hasattr(self, "dataset"):
            self.open_h5()
        label = self.labels[index]
        token_kd = self.token_kds[index]
        cap_gt = self.cap_gts[index]
        cap_gt = [str(i, encoding='utf-8') for i in cap_gt]
        if self.origin=="swin_dert_grid":
            feat = self.feats[index]
            mask = self.masks[index]
            return cap_gt, token_kd, label, feat, mask
        else:
            feat = self.feats[index]
            return cap_gt, token_kd, label, feat
        
    def __len__(self):
        return self.length

def id_coco_imgpath(img_id, is_train=False):
    x = "0"*(12-len(str(img_id)))+str(img_id)+".jpg"
    return x

def id_path(root_path, img_id):
    swin_dert_grid_path = os.path.join(root_path, "feature", "swin_dert_grid", str(img_id)+".npz")
    swin_dert_region_path = os.path.join(root_path, "feature", "swin_dert_region", str(img_id)+".npz")
    image_path = os.path.join(root_path, "feature", "coco2014", "train2014", "COCO_train2014_"+id_coco_imgpath(img_id))
    if not os.path.exists(image_path):
        image_path = os.path.join(root_path, "feature", "coco2014", "val2014", "COCO_val2014_"+id_coco_imgpath(img_id))
    up_down_36_path = os.path.join(root_path, "feature", "up_down_36", str(img_id)+".npz")
    return swin_dert_grid_path, swin_dert_region_path, image_path, up_down_36_path

def get_stop_words(text_field, stop_word_path):
    words = load_txt(stop_word_path)
    stop_word_ids = []
    for w in words:
        stop_word_ids.append(text_field.vocab.stoi[w])
    return stop_word_ids
    
def build_coco_dataloaders(data_path, batch_size, num_workers, origin_cap='transformer', device='cpu'):
    text_field = TextField(vocab_path=os.path.join(data_path, "txt", "coco_vocabulary.txt"), vocab_s_path=os.path.join(data_path, "txt", "se_labels.txt"))
    stop_words = get_stop_words(text_field, os.path.join(data_path, "txt", "english"))
    train_path = os.path.join(data_path, origin_cap+"_train.h5")
    val_path = os.path.join(data_path, origin_cap+"_val.h5")
    test_val_path = os.path.join(data_path, origin_cap+"_val.h5")
    test_test_path = os.path.join(data_path, origin_cap+"_test.h5")
    datasets = {
        'train': PairedDataset(train_path, origin_cap),
        'valid': PairedDataset(val_path, origin_cap),
        'val_test': PairedDataset(test_val_path, origin_cap),
        'test_test': PairedDataset(test_test_path, origin_cap),
    }

    collators = {
        'train': PairedCollator(origin_cap),
        'valid': PairedCollator(origin_cap),
        'val_test': PairedCollator(origin_cap),
        'test_test': PairedCollator(origin_cap),
    }

    batch_size = batch_size
    dataloaders = {}
    dataloaders['train'] = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            collate_fn=collators['train'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=batch_size,
        collate_fn=collators['valid'],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    dataloaders['val_test'] = DataLoader(
        datasets['val_test'],
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collators['val_test'],
        shuffle=False,
        pin_memory=True
    )
    dataloaders['test_test'] = DataLoader(
        datasets['test_test'],
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collators['test_test'],
        shuffle=False,
        pin_memory=True
    )
    return dataloaders, text_field, stop_words

def build_coco_dataloaders_test4w(origin_cap, root_path="../mscoco"):
    test4w_path = os.path.join(root_path, origin_cap+"_test4w.h5")
    text4w = PairedDataset_TEST(test4w_path, origin_cap)
    loader_test4w = DataLoader(
        text4w,
        batch_size=64,
        collate_fn=DictionaryCollator_TEST(origin_cap),
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    
    val3w_path = os.path.join(root_path, origin_cap+"_val3w.h5")
    text3w = PairedDataset_TEST(val3w_path, origin_cap)
    loader_val3w = DataLoader(
        text3w,
        batch_size=64,
        collate_fn=DictionaryCollator_TEST(origin_cap),
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    return loader_test4w, loader_val3w

class PairedDataset_TEST:
    def __init__(self, path, origin):
        self.path = path
        self.origin = origin
        with h5py.File(path, 'r') as f:
            self.length = len(f['image_id'])

    def open_h5(self):
        self.dataset = h5py.File(self.path, 'r')
        self.origin = self.origin
        if self.origin=="swin_dert_grid":
            self.feats = self.dataset['swin_dert_grid']
            self.masks = self.dataset['swin_dert_grid_mask']
        elif self.origin=="up_down_36":
            self.feats = self.dataset['up_down_36']
        self.image_ids = self.dataset['image_id']

    def __getitem__(self, index):
        if not hasattr(self, "dataset"):
            self.open_h5()
        image_id = self.image_ids[index]
        if self.origin=="swin_dert_grid":
            feat = self.feats[index]
            mask = self.masks[index]
            return image_id, feat, mask
        else:
            feat = self.feats[index]
            return image_id, feat
    def __len__(self):
        return self.length

class DictionaryCollator_TEST:
    def __init__(self, origin, device='cpu'):
        self.origin = origin
        self.device = device

    def __call__(self, batch):
        # cap_gt, token_kd, label, feat, mask
        outputs = {}
        image_id = [item[0] for item in batch]
        outputs['image_id'] = image_id

        if self.origin=="swin_dert_grid":
            feat = [item[1] for item in batch]
            feat = torch.from_numpy(np.stack(feat, 0))
            mask = [item[2] for item in batch]
            mask = torch.from_numpy(np.stack(mask, 0))
            outputs['feat'] = feat
            outputs['mask'] = mask
        elif self.origin=="up_down_36":
            feat = [item[1] for item in batch]
            feat = torch.from_numpy(np.stack(feat, 0))
            outputs['feat'] = feat

        return outputs