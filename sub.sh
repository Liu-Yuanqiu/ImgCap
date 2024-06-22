#!/bin/sh
#===========================================================
#配置DSUB资源
#===========================================================
#DSUB --job_type cosched
#DSUB -n s2s_diffusion
#DSUB -A root.dallgdaxrjxylhlsktzu
#DSUB -q root.default
#DSUB -R cpu=32;gpu=1
#DSUB -N 1
#DSUB -oo ./logs/out.log
#DSUB -eo ./logs/err.log

#===========================================================
#加载环境变量
#===========================================================
# conda activate torch-1.11+cu113+py38
python train_s2s.py --exp_name diffusion_loop_test --use_loss_word --use_loss_ce > ./train_diffusion_loop_test.log
