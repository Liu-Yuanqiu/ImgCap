## aic up_dowm_36 xe
Test scores {'BLEU': [0.7638410277048623, 0.6019723134099102, 0.466495935630166, 0.36226434352062653], 'METEOR': 0.2793160702004167, 'ROUGE': 0.565319324640077, 'CIDEr': 1.1520846694129407}

## aic up_dowm_36 scst
Test scores {'BLEU': [0.8161980968857956, 0.6612202545166893, 0.5155474669813525, 0.3947327343853229], 'METEOR': 0.2841276191437215, 'ROUGE': 0.5820427601261694, 'CIDEr': 1.2904751763542661}

## naic up_dowm_36 gt topk20
Test scores {'BLEU': [0.8385001665996755, 0.6746131303175558, 0.5197302972232982, 0.3944789797878372], 'METEOR': 0.28521453546711273, 'ROUGE': 0.5923753337287293, 'CIDEr': 1.2784878137919138}

## naic up_dowm_36 gt topk20 entropy_kv
Test scores {'BLEU': [0.8317468584006258, 0.6683011034061189, 0.5143984464007817, 0.38989753645199204], 'METEOR': 0.2831906679621512, 'ROUGE': 0.589641560202365, 'CIDEr': 1.260837348547825}

## naic up_dowm_36 gt topk10
Test scores {'BLEU': [0.8338287515154018, 0.6720387948891979, 0.5180066631464533, 0.3932937130647021], 'METEOR': 0.28650787570628355, 'ROUGE': 0.5947033342433796, 'CIDEr': 1.2901521335435242}

## naic up_dowm_36 gt topk5

## naic up_dowm_36 diffusion topk10 step10
Test scores {'BLEU': [0.7943555675088795, 0.6285844657518066, 0.47854394146296103, 0.3588236377786171], 'METEOR': 0.267455342317926, 'ROUGE': 0.5652886350662467, 'CIDEr': 1.1577042658293106}


python train_naicdm_step3.py --rank 0 --exp_name step3_up_down_36_topk10_step10_wostep2 --topk 10

python train_naicdm_step3.py --rank 1 --exp_name step3_up_down_36_topk5_step10 --topk 5 --resume_step2

python train_naicdm_step3.py --rank 2 --exp_name step3_up_down_36_topk5_step10_wostep2 --topk 5

python train_naicdm_step3_1.py --rank 2 --exp_name step3_up_down_36_topk5_step10_freeze --topk 5 --resume_step2 --freeze_ei_de



python train_naicdm_step3_h5_layer6.py --rank 1 --exp_name step3_up_down_36_topk10_layer6_step10_bs64 --origin_cap up_down_36 --origin_fea up_down_36 --topk 10 --resume_step2

python train_naicdm_step3_h5_layer6.py --rank 2 --exp_name step3_up_down_36_topk20_layer6_step10_bs64 --origin_cap up_down_36 --origin_fea up_down_36 --topk 20 --resume_step2

3090: step3_up_down_36_topk10_layer6_step10_bs64 (up, bs128) 
    step3_up_down_36_topk20_layer6_step10_bs64 (down, bs128)

309055: step3_swin_dert_grid_topk20_layer6_step10_bs64 (bs128)

wch: step3_swin_dert_grid_topk10_layer6_step10_bs64 (up)
    step3_swin_dert_grid_topk5_layer6_step10_bs64 (down)