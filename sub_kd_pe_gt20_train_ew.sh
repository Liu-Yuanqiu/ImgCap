#!/bin/sh
#===========================================================
#配置DSUB资源
#===========================================================
#DSUB --job_type cosched
#DSUB -n kd_pe_gt20_train_ew
#DSUB -A root.dallgdaxrjxylhlsktzu
#DSUB -q root.default
#DSUB -R cpu=128;gpu=4
#DSUB -N 1
#DSUB -oo ./logs/out_kd_pe_gt20_train_ew.log
#DSUB -eo ./logs/err_kd_pe_gt20_train_ew.log

#===========================================================
#加载环境变量
#===========================================================
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 train_s2s_part.py --exp_name kd_pe_gt20_train_ew --num_timesteps 10 --sample_timesteps 10 --loop 10 --epoch1 100 --epoch2 200 > ./logs/kd_pe_gt20_train_ew.log
# python train_s2s_pe_gt20.py --exp_name kd_pe_gt20_train_ew --epoch1 100 --epoch2 200 > ./logs/kd_pe_gt20.log