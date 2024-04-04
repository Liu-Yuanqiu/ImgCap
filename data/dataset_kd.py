import os
import json
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

class DictionaryCollator:
    def __init__(self, use_cache, device='cpu'):
        self.device = device
        self.use_cache = use_cache

    def __call__(self, batch):
        labels = [item[0] for item in batch]
        labels_out = [item[1] for item in batch]
        caps_kd = [item[2] for item in batch]
        tokens_kd = [item[3] for item in batch]
        caps_gt = [item[4] for item in batch]
        tokens_gt = [item[5] for item in batch]
        image_ids = [item[6] for item in batch]


        outputs = {}
        if self.use_cache:
            grid = [item[7] for item in batch]
            mask = [item[8] for item in batch]
            grid = torch.from_numpy(np.stack(grid, 0)) #.to(self.device)
            mask = torch.from_numpy(np.stack(mask, 0)) #.to(self.device)

            samples = {}
            samples['grid'] = grid
            samples['mask'] = mask
            outputs['samples'] = samples
        else:
            imgs = [item[7] for item in batch]
            outputs['samples'] = nested_tensor_from_tensor_list(imgs) #.to(self.device)

        outputs['labels'] = labels
        outputs['labels_out'] = labels_out
        outputs['caps_kd'] = caps_kd
        outputs['tokens_kd'] = tokens_kd
        outputs['caps_gt'] = caps_gt
        outputs['tokens_gt'] = tokens_gt
        outputs['image_id'] = image_ids
        return outputs

class PairedCollator(DictionaryCollator):

    def __init__(self, use_cache, device='cpu', max_len=54, pad_idx=1, bos_idx=2, eos_idx=3):
        super().__init__(use_cache, device)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.use_cache = use_cache
    # label, cap_kd, token_kd, cap_gt, token_gt, id, grid, mask
    def __call__(self, batch):
        b = super().__call__(batch)

        labels = [l for l in b['labels']]
        labels = torch.from_numpy(np.stack(labels, 0))
        b['labels'] = labels

        labels_out = [l for l in b['labels_out']]
        labels_out = torch.from_numpy(np.stack(labels_out, 0))
        b['labels_out'] = labels_out

        # truncate
        tokens_kd_new = [c[:self.max_len] for c in b['tokens_kd']]
        max_len = max([len(c) for c in b['tokens_kd']])
        # max_len = 20

        padded = []
        for c in tokens_kd_new:
            caption = c + [self.pad_idx] * (max_len - len(c))
            padded.append(caption)

        padded = [torch.Tensor(caption).long() for caption in padded]
        padded = pad_sequence(padded, batch_first=True) #.to(self.device)
        b['tokens_kd'] = padded
        return b
       
class PairedDataset:
    def __init__(self, examples, transform, use_cache, vocab_size):
        self.examples = examples
        self.transform = transform
        self.use_cache = use_cache
        self.onehot = np.identity(vocab_size, dtype=np.int32)
        self.vocab_size = vocab_size
        self.kd_score = 1
        self.gt_score = 1

    def __getitem__(self, index):
        id = self.examples[index]['id']
        filepath = self.examples[index]['image']
        cap_kd = self.examples[index]['cap_kd']
        token_kd = self.examples[index]['token_kd']
        cap_gt = self.examples[index]['cap_gt']
        token_gt = self.examples[index]['token_gt']

        # t_kd_oh = self.onehot[token_kd]
        # t_gt_oh = [self.onehot[t] for t in token_gt]
        # m_l = max([i.shape[0] for i in t_gt_oh])
        # t_gt_oh_new = np.zeros((m_l, self.vocab_size), dtype=np.int32)
        # for t_oh in t_gt_oh:
        #     t_gt_oh_new[:t_oh.shape[0]] += t_oh
        # t_gt_oh = t_gt_oh_new
        # t_gt_oh = np.minimum(t_gt_oh, self.gt_score)
        # max_len = max(t_kd_oh.shape[0], t_gt_oh.shape[0])
        # label = np.zeros((max_len, self.vocab_size))
        # label[:t_kd_oh.shape[0]] = t_kd_oh
        # label[:t_gt_oh.shape[0]] += t_gt_oh
        # label = np.minimum(label, 1)

        max_len = max( max([len(x) for x in token_gt]), len(token_kd) )
        # max_len = 20
        label = np.zeros((self.vocab_size), dtype=np.float32)
        for i in range(max_len):
            for j in range(len(token_gt)):
                if i >= len(token_gt[j]):
                    pass
                else:
                    wid = token_gt[j][i]
                    if wid not in [0, 1, 2, 3]:
                        label[wid] = 1
                    else:
                        pass
        for i in range(max_len):
            if i >= len(token_kd):
                pass
            else:
                wid = token_kd[i]
                if wid not in [0, 1, 2, 3]:
                    label[wid] = 1

        label_out = np.zeros((60, self.vocab_size), dtype=np.float32)
        for i in range(max_len):
            for j in range(len(token_gt)):
                if i >= len(token_gt[j]):
                    pass
                else:
                    wid = token_gt[j][i]
                    if wid not in [0, 1, 2]:
                        label_out[i][wid] = self.gt_score
                    else:
                        pass
        for i in range(max_len):
            if i >= len(token_kd):
                pass
            else:
                wid = token_kd[i]
                if wid not in [0, 1, 2]:
                    label_out[i][wid] = self.kd_score

        if self.use_cache:
            with np.load(filepath, allow_pickle=True) as data_grid:
                grid = data_grid['grid']
                grid = np.array(grid).astype('float32')
                mask = data_grid['mask']
                mask = np.array(mask).astype('bool')
            return label, label_out, cap_kd, token_kd, cap_gt, token_gt, id, grid, mask
        else:
            img = Image.open(filepath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return label, label_out, cap_kd, token_kd, cap_kd, token_gt, id, img
        
    def __len__(self):
        return len(self.examples)

def id_coco_imgpath(img_id, is_train=False):
    x = "0"*(12-len(str(img_id)))+str(img_id)+".jpg"
    return x

def id_path(root_path, img_id, use_cache):
    if use_cache:
        return os.path.join(root_path, "feature", "swin_dert_grid", str(img_id)+".npz")
    else:
        train_path = os.path.join(root_path, "feature", "coco2014", "COCO_train2014_"+id_coco_imgpath(img_id))
        if os.path.exists(train_path):
            return train_path
        else:
            return os.path.join(root_path, "feature", "coco2014", "COCO_val2014_"+id_coco_imgpath(img_id))

class COCO_KD:
    def __init__(self, text_field, root_path, use_cache):
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        if os.path.exists(os.path.join(root_path, "cached_coco_train.json")):
            self.train_samples = json.load(open(os.path.join(root_path, "cached_coco_train.json"), "r"))
            self.val_samples = json.load(open(os.path.join(root_path, "cached_coco_val.json"), "r"))
            self.test_samples = json.load(open(os.path.join(root_path, "cached_coco_test.json"), "r"))
        else:
            self.text_field = text_field
            ids_train = load_txt(os.path.join(root_path, 'txt', 'coco_train_image_id.txt'))
            ids_val = load_txt(os.path.join(root_path, 'txt', 'coco_val_image_id.txt'))
            ids_test = load_txt(os.path.join(root_path, 'txt', 'coco_test_image_id.txt'))
            
            samples = json.load(open(os.path.join(root_path, "annotations", 'captions_kd3.json'), "r"))

            for sam in samples:
                img_id = sam['image_id']
                cap_kd = sam['gen']
                cap_gt = sam['gts']
                
                filepath = id_path(root_path, img_id, use_cache)
                token_kd = [self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(cap_kd)]
                token_kd = token_kd + [text_field.vocab.stoi['<eos>']]
                token_gt = [[self.text_field.vocab.stoi[w] for w in t] for t in self.text_field.preprocess(cap_gt)]
                token_gt = [t+[text_field.vocab.stoi['<eos>']] for t in token_gt]
                s = {"id":img_id, "image": filepath, "cap_kd":cap_kd, "token_kd": token_kd, "cap_gt":cap_gt, "token_gt": token_gt}
                
                if img_id in ids_train:
                    self.train_samples.append(s)
                elif img_id in ids_val:
                    self.val_samples.append(s)
                elif img_id in ids_test:
                    self.test_samples.append(s)
                else:
                    raise ValueError("wrong image id")
            json.dump(self.train_samples, open(os.path.join(root_path, "cached_coco_train.json"), "w"))
            json.dump(self.val_samples, open(os.path.join(root_path, "cached_coco_val.json"), "w"))
            json.dump(self.test_samples, open(os.path.join(root_path, "cached_coco_test.json"), "w"))
            
def build_coco_dataloaders(config=None, device='cpu'):
    transform = get_transform(config.dataset.transform)

    use_cache = config.dataset.use_cache
    text_field = TextField(vocab_path=config.dataset.vocab_path)
    coco = COCO_KD(text_field, config.dataset.root_path, use_cache)

    datasets = {
        'train': PairedDataset(coco.train_samples, transform['train'], use_cache, len(text_field.vocab)),
        'valid': PairedDataset(coco.val_samples, transform['valid'], use_cache, len(text_field.vocab)),
        'test': PairedDataset(coco.test_samples, transform['valid'], use_cache, len(text_field.vocab)),
    }
    # label = datasets['train'].__getitem__(120)[6]
    # for i in range(label.shape[0]):
    #     l = label[i]
    #     print(l.sum(), end=" ")
    # print(label.shape)
    collators = {
        'train': PairedCollator(use_cache, device=device),
        'valid': PairedCollator(use_cache, device=device),
        'test': PairedCollator(use_cache, device=device),
    }

    batch_size = config.optimizer.batch_size
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        collate_fn=collators['train'],
        num_workers=config.optimizer.num_workers,
        shuffle=False,
        pin_memory=True
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=batch_size,
        collate_fn=collators['valid'],
        num_workers=config.optimizer.num_workers,
        shuffle=False,
        pin_memory=True
    )
    dataloaders['test'] = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['test'],
        shuffle=False,
        pin_memory=True
    )
    return dataloaders, text_field