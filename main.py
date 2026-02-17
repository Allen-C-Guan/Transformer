

import torch
import torch.nn as nn
from torch.utils.backcompat import keepdim_warning

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    t = torch.triu(torch.Tensor([[1,2],[3,4]]), 1)
    print(t)