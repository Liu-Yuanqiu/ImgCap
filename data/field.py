# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
from itertools import chain
import six
import torch
import numpy as np
import cv2
import h5py
import os
import warnings
import shutil
import pickle
import time
import itertools

from .vocab import Vocab
from .utils import get_tokenizer
from torchvision.transforms import ToTensor

class TextField:
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token="<bos>", eos_token="<eos>", fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=True, tokenize="spacy",
                 remove_punctuation=True, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True, vocab_path=None, vocab_s_path=None):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = Vocab(vocab_path=vocab_path, tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
        # self.vocab_s = Vocab(vocab_path=vocab_s_path, tokens=["<unk>"])
        if nopoints:
            self.punctuations.append("..")

    def preprocess(self, x):
        if isinstance(x, str):
            if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
                x = six.text_type(x, encoding='utf-8')
            if self.lower:
                x = six.text_type.lower(x)
            x = self.tokenize(x.rstrip('\n'))
            if self.remove_punctuation:
                x = [w for w in x if w not in self.punctuations]
            if self.preprocessing is not None:
                return self.preprocessing(x)
            else:
                return x
        elif isinstance(x, list):
            xn = []
            for xx in x:
                if six.PY2 and isinstance(xx, six.string_types) and not isinstance(xx, six.text_type):
                    xx = six.text_type(xx, encoding='utf-8')
                if self.lower:
                    xx = six.text_type.lower(xx)
                xx = self.tokenize(xx.rstrip('\n'))
                if self.remove_punctuation:
                    xx = [w for w in xx if w not in self.punctuations]
                if self.preprocessing is not None:
                    xn.append(self.preprocessing(xx))
                else:
                    xn.append(xx)
            return xn
        else:
            raise ValueError("input data must be one of (str, list)")

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
                
            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True, deduplication=False):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words, deduplication)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words, deduplication)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words, deduplication)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words, deduplication)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.unk_token:
                    continue
                if word == self.eos_token:
                    break
                caption.append(word)
            
            if deduplication:
                # for i in range(len(caption)-1, 0, -1):
                #     if caption[i] == caption[i-1]:
                #         del caption[i]
                caption = [k for k, g in itertools.groupby(caption)]

            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions
