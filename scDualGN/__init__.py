#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import torch


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()
torch.cuda.empty_cache()

from .nets import *
from .run import scDualGN
from .evalution import *
from .utils import data_preprocess
