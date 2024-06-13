import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.s2s.transformer_word import Transformer
# from models.losses import FocalLossWithLogitsNegLoss, WeightedFocalLossWithLogitsNegLoss
from models.metric import MultiLabelAccuracy, mAPMeter
from pycocotools.coco import COCO
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.nn import NLLLoss, MSELoss
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from omegaconf import OmegaConf

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def normal_word():
    import json
    all_data = []
    data = json.load(open("/home/liuyuanqiu/code/mscoco/annotations/captions_train2014.json", "r"))
    all_data += data['annotations']
    data = json.load(open("/home/liuyuanqiu/code/mscoco/annotations/captions_val2014.json", "r"))
    all_data += data['annotations']
    word_count = {}
    for d in all_data:
        cap = d['caption']
        words = cap.lower().split()
        for w in words:
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1
    sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_word_count)
    mormal_ws = {}
    normal_w = []
    for k,v in sorted_word_count.items():
        if v>10000:
            mormal_ws[k] = v
            normal_w.append(k)
    print(mormal_ws)
    print(normal_w)

if __name__ == '__main__':
    # normal_word()
    # args = OmegaConf.load('configs/s2s_word.yaml')
    # print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.rank
    # device = torch.device('cuda')
    # multiprocessing.set_start_method('spawn')

    # writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.mode, args.exp_name))

    # dataloaders, text_field = build_coco_dataloaders(args, device)
    
    # normal_w = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'man', \
    #             'to', 'an', 'two', 'at', 'are', 'people', \
    #             'next', 'woman', 'that',  'some', 'large', \
    #             'person', 'down',  'top', 'up',  'small', 'near', \
    #             'his',  'front', 'by', 'has', 'while',  'it.', 'there',  \
    #             'three', 'for',  'it', 'boy', 'men', 'other']
    # wid = []
    # for w in normal_w:
    #     wid.append(text_field.vocab.stoi[w])
    # print(wid)
    fname = os.path.join("./ckpts", "s2s", "test", "s2s_best.pth")
    data = torch.load(fname)['state_dict']
    for k in data:
        print(k)