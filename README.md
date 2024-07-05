# A PyTorch Implementation For Image Captioning

## Installation
1. Download spacy data by executing the following command:
```
python -m spacy download en_core_web_sm
```

2. Add evaluation module from [evaluation](https://github.com/aimagelab/meshed-memory-transformer/tree/master/evaluation).

Note: Python 3.6+ and Pytorch 1.6+ are required to run our code. 

3. Install Deformable Attention:
```shell
cd models/detector/ops/
python setup.py build develop
python test.py
```

## diffusion_loop_test1
```
use kd; step100; sample100; loop10
python train_s2s.py --exp_name diffusion_loop_test1 --epoch1 43 --epoch2 45

Epoch: 56, Learning Rate: 0.000004
Validation scores {'BLEU': [0.8135049920187617, 0.6554888589426111, 0.5058054141056576, 0.38364798186371213], 'METEOR': 0.28121353295191004, 'ROUGE': 0.5810628828015832, 'CIDEr': 1.2556216017448463}
Test scores {'BLEU': [0.8145345778453518, 0.6546796721911093, 0.5044087463643228, 0.38201691914476105], 'METEOR': 0.2799974473714809, 'ROUGE': 0.5786321355001061, 'CIDEr': 1.2619082192175866}
```
## kd_step100_sample100_loop10
```
python train_s2s.py --rank 3 --exp_name kd_step100_sample100_loop10 --num_timesteps 100 --sample_timesteps 100 --loop 10 --epoch1 100 --epoch2 200
```
## kd_step100_sample10_loop10
```
python train_s2s.py --rank 2 --exp_name kd_step100_sample10_loop10 --num_timesteps 100 --sample_timesteps 10 --loop 10 --epoch1 100 --epoch2 200
```
## kd_step10_sample10_loop10
```
python train_s2s.py  --rank 1 --exp_name kd_step10_sample10_loop10 --num_timesteps 10 --sample_timesteps 10 --loop 10 --epoch1 100 --epoch2 200
```
## kd3_step100_sample100_loop10
```
python train_s2s.py --exp_name kd3_step100_sample100_loop10 --num_timesteps 100 --sample_timesteps 100 --loop 10 --epoch1 100 --epoch2 200
```
## kd3_step100_sample10_loop10
```
python train_s2s.py --exp_name kd3_step100_sample10_loop10 --num_timesteps 100 --sample_timesteps 10 --loop 10 --epoch1 100 --epoch2 200
```

## kd_step10_sample10_loop10_new
```
训练过程中从v得到初始值，作为outw生成句子
python train_s2s.py --rank 1 --exp_name kd_step10_sample10_loop10_new --num_timesteps 10 --sample_timesteps 10 --loop 10 --epoch1 100 --epoch2 200
```

## img_kd_step10_sample10_loop10
```
outi和outw都从扩散网络中得到
python train_e2es2s.py --rank 2 --exp_name img_kd_step10_sample10_loop10 --num_timesteps 10 --sample_timesteps 10 --loop 10 --epoch1 100 --epoch2 200
```
## Acknowledge
This repo is based on [M^2 Transformer](https://github.com/aimagelab/meshed-memory-transformer), [the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
