import torch
# import torch.nn as nn
# from torch.utils.backcompat import keepdim_warning

'''
维度，长度，和操作的关系
选择一个维度的时候，内再的含义是，其他维度都已经确定下，只有这个维度的值取了全集。而这个全集带来的含义，就是这个维度的含义！操作也是操作这个维度。
'''
if __name__ == '__main__':
    l = torch.tensor( [[[1,2, 3,4],[5, 6,7,8], [9,10,11,12]], [[13,14,16,17],[18,19,20,21],[22,23,24,25]]])
    # print(l)
    # print(l.size())
    # print(l.shape)
    #

    l2 = torch.tensor([[1,2,3,4], [5,6,7,8]])
    mask = torch.tensor([True,False])
    print(l2[mask])
    print(l2[mask, 2])

    # print(l.topk(3,-1))
    # print(torch.full_like(l, 1000))
    #
    # l1 = torch.tensor([[1,2,3,4],
    #       [5,6,7,8],
    #       [9,10,11,12]])
    # src_index = torch.tensor([[3,2,1,0],
    #              [1,1,1,1],
    #              [0,0,0,0]])
    #
    # l2 = torch.zeros(5,4, dtype = int)
    # l2  = l2.scatter(1, src_index, l1)
    # print(l2)
