import os
import numpy as np
import itertools
import collections
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from .utils import load_txt, nested_tensor_from_tensor_list
from .transforms import get_transform
from .field import TextField
from pycocotools.coco import COCO as pyCOCO

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class DictionaryCollator:
    def __init__(self, use_cache, device='cpu'):
        self.device = device
        self.use_cache = use_cache

    def __call__(self, batch):
        captions = [item[0] for item in batch]
        image_ids = [item[1] for item in batch]

        outputs = {}
        if self.use_cache:
            grid = [item[2] for item in batch]
            mask = [item[3] for item in batch]
            grid = torch.from_numpy(np.stack(grid, 0))
            mask = torch.from_numpy(np.stack(mask, 0))
            samples = {}
            samples['grid'] = grid
            samples['mask'] = mask
            outputs['samples'] = samples
            # grid = [item[2] for item in batch]
            # grid = torch.from_numpy(np.stack(grid, 0))
            # samples = {}
            # samples['grid'] = grid
            # samples['mask'] = None
            # outputs['samples'] = samples
        else:
            imgs = [item[2] for item in batch]
            outputs['samples'] = torch.from_numpy(np.stack(imgs, 0))

        outputs['captions'] = captions
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

    def __call__(self, batch):
        b = super().__call__(batch)

        # truncate
        captions = [c[:self.max_len] for c in b['captions']]
        max_len = max([len(c) for c in b['captions']])

        padded = []
        for c in captions:
            caption = [self.bos_idx] + c + [self.eos_idx] + [self.pad_idx] * (max_len - len(c))
            padded.append(caption)

        padded = [torch.Tensor(caption).long() for caption in padded]
        padded = pad_sequence(padded, batch_first=True, padding_value=self.pad_idx)

        b['captions'] = padded
        return b

class DictionaryDataset:
    def __init__(self, examples, transform, use_cache):
        self.examples = examples
        self.transform = transform
        self.use_cache = use_cache

    def __getitem__(self, index):
        id = self.examples[index]['id']
        filepath = self.examples[index]['image']
        text = self.examples[index]['text']
        caption = text

        if self.use_cache:
            img = {}
            with np.load(filepath, allow_pickle=True) as data_grid:
                grid = data_grid['grid']
                grid = np.array(grid).astype('float32')
                mask = data_grid['mask']
                mask = np.array(mask).astype('bool')
            return caption, id, grid, mask
            # with np.load(filepath, allow_pickle=True) as data:
            #     region_ud = data['feat']
            #     region_ud = np.array(region_ud).astype('float32')
            # return caption, id, region_ud
        else:
            img = Image.open(filepath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return caption, id, img
    
    def __len__(self):
        return len(self.examples)
        
class PairedDataset:
    def __init__(self, examples, transform, use_cache):
        self.examples = examples
        self.transform = transform
        self.use_cache = use_cache

    def __getitem__(self, index):
        id = self.examples[index]['id']
        filepath = self.examples[index]['image']
        text = self.examples[index]['text']
        token = self.examples[index]['token']
        caption = token

        if self.use_cache:
            img = {}
            with np.load(filepath, allow_pickle=True) as data_grid:
                grid = data_grid['grid']
                grid = np.array(grid).astype('float32')
                mask = data_grid['mask']
                mask = np.array(mask).astype('bool')
            return caption, id, grid, mask
            # with np.load(filepath, allow_pickle=True) as data:
            #     region_ud = data['feat']
            #     region_ud = np.array(region_ud).astype('float32')
            # return caption, id, region_ud
        else:
            img = Image.open(filepath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return caption, id, img
        
    def __len__(self):
        return len(self.examples)

class COCO:
    def __init__(self, text_field, root_path, use_cache):
        self.text_field = text_field
        ids_train = load_txt(os.path.join(root_path, 'txt', 'coco_train_image_id.txt'))
        ids_val = load_txt(os.path.join(root_path, 'txt', 'coco_val_image_id.txt'))
        ids_test = load_txt(os.path.join(root_path, 'txt', 'coco_test_image_id.txt'))
        self.train_samples = []
        self.val_samples = []
        self.train_dict_samples = []
        self.val_dict_samples = []
        self.test_dict_samples = []

        dataset_train2014 = pyCOCO(os.path.join(root_path, 'annotations', 'captions_train2014.json'))
        dataset_val2014 = pyCOCO(os.path.join(root_path, 'annotations', 'captions_val2014.json'))
        for id_train in ids_train:
            if id_train in dataset_train2014.imgToAnns.keys():
                anns = dataset_train2014.imgToAnns[id_train]
                anns = [an["caption"] for an in anns]
                if use_cache:
                    filepath = os.path.join(root_path, "feature", "swin_dert_grid", str(id_train)+".npz")
                    # filepath = os.path.join(root_path, "feature", "up_down_36", str(id_train)+".npz")
                else:
                    filename = "COCO_train2014_"+"0"*(12-len(str(id_train)))+str(id_train)+".jpg"
                    filepath = os.path.join(root_path, "feature", "coco2014", "train2014", filename)
                self.train_dict_samples.append({"id":id_train, "image": filepath, "text":anns})
                for ann in anns:
                    token = [self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(ann)]
                    self.train_samples.append({"id":id_train, "image": filepath, "text":ann, "token": token})
            else:
                anns = dataset_val2014.imgToAnns[id_train]
                anns = [an["caption"] for an in anns]
                if use_cache:
                    filepath = os.path.join(root_path, "feature", "swin_dert_grid", str(id_train)+".npz")
                    # filepath = os.path.join(root_path, "feature", "up_down_36", str(id_train)+".npz")
                else:
                    filename = "COCO_val2014_"+"0"*(12-len(str(id_train)))+str(id_train)+".jpg"
                    filepath = os.path.join(root_path, "feature", "coco2014", "val2014", filename)
                self.train_dict_samples.append({"id":id_train, "image": filepath, "text":anns})
                for ann in anns:
                    token = [self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(ann)]
                    self.train_samples.append({"id":id_train, "image": filepath, "text":ann, "token": token})
        
        for id_val in ids_val:
            anns = dataset_val2014.imgToAnns[id_val]
            anns = [an["caption"] for an in anns]
            if use_cache:
                filepath = os.path.join(root_path, "feature", "swin_dert_grid", str(id_val)+".npz")
                # filepath = os.path.join(root_path, "feature", "up_down_36", str(id_val)+".npz")
            else:
                filename = "COCO_val2014_"+"0"*(12-len(str(id_val)))+str(id_val)+".jpg"
                filepath = os.path.join(root_path, "feature", "coco2014", "val2014", filename)
            self.val_dict_samples.append({"id":id_val, "image": filepath, "text":anns})
            for ann in anns:
                token = [self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(ann)]
                self.val_samples.append({"id":id_val, "image": filepath, "text":ann, "token": token})

        for id_test in ids_test:
            anns = dataset_val2014.imgToAnns[id_test]
            anns = [an["caption"] for an in anns]
            if use_cache:
                filepath = os.path.join(root_path, "feature", "swin_dert_grid", str(id_test)+".npz")
                # filepath = os.path.join(root_path, "feature", "up_down_36", str(id_test)+".npz")
            else:
                filename = "COCO_val2014_"+"0"*(12-len(str(id_test)))+str(id_test)+".jpg"
                filepath = os.path.join(root_path, "feature", "coco2014", "val2014", filename)
            self.test_dict_samples.append({"id":id_test, "image": filepath, "text":anns})


def build_coco_dataloaders(config=None, device='cpu'):
    # transform = get_transform(config.dataset.transform)
    transform = {}
    transform["train"] = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
    transform["valid"] = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    use_cache = config.dataset.use_cache
    text_field = TextField(vocab_path=config.dataset.vocab_path)
    coco = COCO(text_field, config.dataset.root_path, use_cache)

    datasets = {
        'train': PairedDataset(coco.train_samples, transform['train'], use_cache),
        'valid': PairedDataset(coco.val_samples, transform['valid'], use_cache),
        'train_dict': DictionaryDataset(coco.train_dict_samples, transform['train'], use_cache),
        'valid_dict': DictionaryDataset(coco.val_dict_samples, transform['valid'], use_cache),
        'test_dict': DictionaryDataset(coco.test_dict_samples, transform['valid'], use_cache),
    }

    collators = {
        'train': PairedCollator(use_cache, device=device),
        'valid': PairedCollator(use_cache, device=device),
        'train_dict': DictionaryCollator(use_cache, device=device),
        'valid_dict': DictionaryCollator(use_cache, device=device),
        'test_dict': DictionaryCollator(use_cache, device=device),
    }

    batch_size = config.optimizer.batch_size
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        collate_fn=collators['train'],
        num_workers=config.optimizer.num_workers,
        shuffle=True
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=batch_size,
        collate_fn=collators['valid'],
        num_workers=config.optimizer.num_workers,
        shuffle=False
    )
    dataloaders['train_dict'] = DataLoader(
        datasets['train_dict'],
        batch_size=batch_size,
        collate_fn=collators['train_dict'],
        num_workers=config.optimizer.num_workers,
        shuffle=True
    )
    dataloaders['valid_dict'] = DataLoader(
        datasets['valid_dict'],
        batch_size=batch_size,
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['valid_dict'],
        shuffle=False
    )
    dataloaders['test_dict'] = DataLoader(
        datasets['test_dict'],
        batch_size=batch_size,
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['test_dict'],
        shuffle=False
    )
    return dataloaders, text_field