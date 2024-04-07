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
### s2s
- ed: 使用三个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。label使用蒸馏后句子和真实句子所有单词，有则为1，没有为0。交叉熵损失。

- ed: test分支。与ed相同，讲模型结构改回使用word embedding。

- ed_labelweught: 使用三个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。label使用蒸馏后句子和真实句子所有单词，出现一次加一。交叉熵损失。

- ed_xe: 使用三个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。仅使用交叉熵损失。

- ed_1: 使用一个模型的蒸馏结果训练。三层编码器图像特征，三层编码器生成单词，六层解码器生成最终描述。label使用蒸馏后句子和真实句子所有单词，有则为1，没有为0。交叉熵损失。


## Acknowledge
This repo is based on [M^2 Transformer](https://github.com/aimagelab/meshed-memory-transformer), [the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
