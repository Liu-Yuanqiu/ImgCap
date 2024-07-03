#!/bin/sh
#===========================================================
#配置DSUB资源
#===========================================================
#DSUB --job_type cosched
#DSUB -n kd3_step100_sample100_loop10
#DSUB -A root.dallgdaxrjxylhlsktzu
#DSUB -q root.default
#DSUB -R cpu=32;gpu=1
#DSUB -N 1
#DSUB -oo ./logs/out_kd3_step100_sample100_loop10.log
#DSUB -eo ./logs/err_kd3_step100_sample100_loop10.log

#===========================================================
#加载环境变量
#===========================================================
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 train_s2s_ddp.py --exp_name kd3_step100_sample100_loop10 --num_timesteps 100 --sample_timesteps 100 --loop 10 --epoch1 100 --epoch2 200 > ./logs/kd3_step100_sample100_loop10.log
# python train_s2s.py --exp_name kd3_step100_sample100_loop10 --num_timesteps 100 --sample_timesteps 100 --loop 10 --epoch1 100 --epoch2 200 > ./logs/kd3_step100_sample100_loop10.log