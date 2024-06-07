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

## Training procedure
Run `python train.py` using the following arguments:

| Argument | Model |
|------|------|
| `python train_transformer.py` | Auto-regressive Image Captioning |

## 实验记录
### teacher model
- tm_gt20_ce: 基础版本，使用真实20单词，交叉熵监督训练
```
python train_s2s.py --rank 0 --exp_name tm_gt20_ce --use_loss_ce
```
- tm_gt20_ce_entropy：在基础版本上增加熵损失
```
python train_s2s.py --rank 1 --exp_name tm_gt20_ce_entropy --use_loss_ce --use_loss_entropy
```
- tm_gt20_ce_l2：在基础版本上增加l2损失，控制word_emb靠近0
```
python train_s2s.py --rank 0 --exp_name tm_gt20_ce --use_loss_ce --use_loss_kl
```
- tm_gt20_ce_l2_entropy：全量版本
```
python train_s2s.py --rank 0 --exp_name tm_gt20_ce --use_loss_ce --use_loss_kl --use_loss_entropy
```
### s2s
- ed: 使用三个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。label使用蒸馏后句子和真实句子所有单词，有则为1，没有为0。交叉熵损失。

- ed_labelweught: 使用三个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。label使用蒸馏后句子和真实句子所有单词，出现一次加一。交叉熵损失。

- ed_xe: 使用三个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。仅使用交叉熵损失。

- ed_1: 使用一个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。label使用蒸馏后句子和真实句子所有单词，有则为1，没有为0。交叉熵损失。


## Acknowledge
This repo is based on [M^2 Transformer](https://github.com/aimagelab/meshed-memory-transformer), [the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
