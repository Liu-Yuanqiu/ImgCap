#!/bin/sh
#===========================================================
#配置DSUB资源
#===========================================================
#DSUB --job_type cosched
#DSUB -n s2s_diffusion
#DSUB -A root.dallgdaxrjxylhlsktzu
#DSUB -q root.default
#DSUB -R cpu=128;gpu=4
#DSUB -N 1
#DSUB -oo ./logs/out.log
#DSUB -eo ./logs/err.log

#===========================================================
#加载环境变量
#===========================================================
# conda activate torch-1.11+cu113+py38
# python train_s2s.py --exp_name diffusion_loop_test1  > ./train_diffusion_loop_test1.log
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 train_s2s_ddp.py --exp_name diffusion_loop_test4 > ./train_diffusion_loop_test4.log