#!/bin/sh
#===========================================================
#配置DSUB资源
#===========================================================
#DSUB --job_type cosched
#DSUB -n transformer_sdg_bs96_w12_sample50_loop10
#DSUB -A root.dallgdaxrjxylhlsktzu
#DSUB -q root.default
#DSUB -R cpu=32;gpu=1
#DSUB -N 1
#DSUB -oo ./logs/transformer_sdg_bs96_w12_sample50_loop10.log
#DSUB -eo ./logs/transformer_sdg_bs96_w12_sample50_loop10.log

#===========================================================
#加载环境变量
#===========================================================
python train_s2s.py --exp_name transformer_sdg_bs96_w12_sample50_loop10 --origin_cap transformer --origin_fea swin_dert_grid --sample_timesteps 50 --loop 10 --epoch1 100 --epoch2 200 > ./logs/transformer_sdg_bs96_w12_sample50_loop10.log