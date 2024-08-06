import random
import evaluation
from evaluation import Cider
from data.dataset_kd_h5 import build_coco_dataloaders, build_coco_dataloaders_test4w
from models.naicdm.naicdm_layer6 import Transformer
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            cap_gt = batch['cap_gt']
            if origin_fea == "swin_dert_grid":
                feat, mask = batch["feat"].to(device), batch["mask"].to(device)
            elif origin_fea == "swin_dert_region":
                feat, mask = batch["feat"].to(device), batch["mask"].to(device)
            elif origin_fea == "up_down_36":
                feat, mask = batch["feat"].to(device), None
            else:
                raise NotImplementedError

            with torch.no_grad():
                logit = model.infer(feat, mask)
            
            _, out = torch.max(logit, -1)
            caps_gen = text_field.decode(out, join_words=False, deduplication=False)
            for i, (gts_i, gen_i) in enumerate(zip(cap_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def infer(model, dataloader, text_field):
    import itertools
    model.eval()
    data = []
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id = batch['image_id']
            if origin_fea == "swin_dert_grid":
                feat, mask = batch["feat"].to(device), batch["mask"].to(device)
            elif origin_fea == "swin_dert_region":
                feat, mask = batch["feat"].to(device), batch["mask"].to(device)
            elif origin_fea == "up_down_36":
                feat, mask = batch["feat"].to(device), None
            else:
                raise NotImplementedError

            with torch.no_grad():
                logit = model.infer(feat, mask)
            
            _, out = torch.max(logit, -1)
            caps_gen = text_field.decode(out, join_words=True, deduplication=True)
            for i, (id, gen_i) in enumerate(zip(image_id, caps_gen)):
                d = {}
                d['image_id'] = int(id)
                d['caption'] = gen_i
                data.append(d)
            pbar.update()
    return data

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda')
    multiprocessing.set_start_method('spawn')
    origin_cap = "swin_dert_grid"
    origin_fea = "swin_dert_grid"
    if origin_fea == "swin_dert_grid":
        feat_dim = 1024
    elif origin_fea == "swin_dert_region":
        feat_dim = 512
    elif origin_fea == "up_down_36":
        feat_dim = 2048
    else:
        raise NotImplementedError
    topk = 5
    layer_num = 6
    step = 1
    root_path = "../mscoco"
    loaders, text_field, stop_words = build_coco_dataloaders(root_path, 64, 4, origin_cap)
    loader4w, load3w = build_coco_dataloaders_test4w(origin_cap)

    # Model and dataloaders
    model = Transformer(feat_dim, len(text_field.vocab), text_field.vocab.stoi['<pad>'], \
                        topk, num_timesteps=step, sampling_timesteps=step,\
                        N_en=layer_num, N_wo=layer_num, N_de=layer_num).to(device)
    model.tensor_to(device)
    model_path = os.path.join("./ckpts", "naicdm", "step3_swin_dert_grid_topk5_layer6_step10_bs64")
    fname = os.path.join(model_path, '%s_best.pth' % "naicdm")
    assert os.path.exists(fname), "weight is not found"
    data = torch.load(fname)
    # torch.set_rng_state(data['torch_rng_state'])
    # torch.cuda.set_rng_state(data['cuda_rng_state'])
    # np.random.set_state(data['numpy_rng_state'])
    # random.setstate(data['random_rng_state'])
    model.load_state_dict(data['state_dict'], strict=False)
    print('Resuming from epoch %d, best cider %f' % (
                data['epoch'], data['best_cider']))
    model.eval()
    # scores = evaluate_metrics(model, loaders["val_test"], text_field)
    # print("Validation scores", scores)
    # scores = evaluate_metrics(model, loaders["test_test"], text_field)
    # print("Test scores", scores)
    
    start = time.time()
    data4w = infer(model, loader4w, text_field)
    all_time = time.time() - start
    print("per image: %f" % (all_time/len(loader4w)))
    with open("./captions_test2014_results_"+origin_cap+"1.json", "w") as f:
        json.dump(data4w, f)

    start = time.time()
    data3w = infer(model, load3w, text_field)
    all_time = time.time() - start
    print("per image: %f" % (all_time/len(load3w)))
    with open("./captions_val2014_results_"+origin_cap+"1.json", "w") as f:
        json.dump(data3w, f)