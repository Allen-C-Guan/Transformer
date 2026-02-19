import torch
# import torch.nn as nn
# from torch.utils.backcompat import keepdim_warning

'''
维度，长度，和操作的关系
选择一个维度的时候，内再的含义是，其他维度都已经确定下，只有这个维度的值取了全集。而这个全集带来的含义，就是这个维度的含义！操作也是操作这个维度。
'''
if __name__ == '__main__':
    l = torch.tensor( [[1,2],[3,4]])
    l = l.sum(dim = -1)
    print(l)
