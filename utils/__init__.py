from .utils import download_from_url
from data.utils import NestedTensor
from .typing import *

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    elif isinstance(x, NestedTensor):
        b_s = x.tensors.size(0)
    elif isinstance(x, dict):
        b_s = list(x.values())[0].size(0)
    else:
        b_s = x[0].size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    elif isinstance(x, NestedTensor):
        b_s = x.tensors.device
    elif isinstance(x, dict):
        b_s = list(x.values())[0].device
    else:
        b_s = x[0].device
    return b_s
