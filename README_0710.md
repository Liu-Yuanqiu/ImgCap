## gt20_kd
```
训练编码器和解码器，得到一个比较好的解码器权重
python train_s2s_pe_gt20.py --rank 0 --exp_name gt20_kd --sample_timesteps 100 --loop 1 --epoch1 100 --epoch2 200
```
## kd_sample100
```
冻结编码器和解码器部分，训练单词预测网络
CUDA_VISIBLE_DEVICES="3" python -m torch.distributed.launch --nproc_per_node 1 train_s2s_part.py --exp_name kd_sample100_test --sample_timesteps 100 --loop 10 --epoch1 100 --epoch2 200
```
## 071221
```
使用clip检索句子，句子得到的单词与真实单词做交叉熵损失，然后位置编码作为K，单词向量作为QV，生成句子。
CUDA_VISIBLE_DEVICES="2" python train_ranaic.py --exp_name 071221 --clip_model_path ViT-B/32 --origin_cap transformer --origin_fea swin_dert_grid --batch_size 64 --workers 8 --loop 10 --epoch1 100 --epoch2 100
```
## 071223UDR
```
使用clip检索句子，句子得到的单词与真实单词做交叉熵损失，然后位置编码作为K，单词向量作为QV，生成句子。
CUDA_VISIBLE_DEVICES="1" python train_ranaic1.py --exp_name 071223UDR --clip_model_path ViT-B/32 --origin_cap transformer --origin_fea up_down_36 --batch_size 64 --workers 8 --loop 10 --epoch1 100 --epoch2 100
```
## 071223SDR
```
使用clip检索句子，句子得到的单词与真实单词做交叉熵损失，然后位置编码作为K，单词向量作为QV，生成句子。
CUDA_VISIBLE_DEVICES="2" python train_ranaic1.py --exp_name 071223SDR --clip_model_path ViT-B/32 --origin_cap transformer --origin_fea swin_dert_region --batch_size 64 --workers 8 --loop 10 --epoch1 100 --epoch2 100
```
## 071223SDG
```
使用clip检索句子，句子得到的单词与真实单词做交叉熵损失，然后位置编码作为K，单词向量作为QV，生成句子。
CUDA_VISIBLE_DEVICES="3" python train_ranaic1.py --exp_name 071223SDG --clip_model_path ViT-B/32 --origin_cap transformer --origin_fea swin_dert_grid --batch_size 64 --workers 8 --loop 10 --epoch1 100 --epoch2 100
```

## Acknowledge
This repo is based on [M^2 Transformer](https://github.com/aimagelab/meshed-memory-transformer), [the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
