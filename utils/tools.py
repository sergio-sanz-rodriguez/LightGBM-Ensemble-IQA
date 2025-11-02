import torch
import random
import numpy as np
import os


def setup_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def seed_torch(seed):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    # torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速