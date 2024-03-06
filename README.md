# A PyTorch Implementation For Image Captioning

## Installation
1. Download spacy data by executing the following command:
```
python -m spacy download en
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


## Acknowledge
This repo is based on [M^2 Transformer](https://github.com/aimagelab/meshed-memory-transformer), [the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
