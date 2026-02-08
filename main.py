

import torch
import torch.nn as nn
from torch.utils.backcompat import keepdim_warning

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    import torch

    # 创建一个 1x12 的张量
    x = torch.arange(12)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    print("原始张量:", x)
    print("原始形状:", x.shape)  # torch.Size([12])

    # 使用 view 改变形状
    y = x.view(3, 4)  # 改为 3x4
    print("\nview(3, 4) 后:")
    print(y)
    print("形状:", y.shape)  # torch.Size([3, 4])
    # 输出:
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])

    # 改为三维
    z = x.view(2, 2, 3)  # 改为 2x2x3
    print("\nview(2, 2, 3) 后:")
    print(z)
    print("形状:", z.shape)  # torch.Size([2, 2, 3])


    print("end of main")
