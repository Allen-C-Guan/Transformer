import torch
from torch import nn

class Expert(nn.Module):
    '''
    输入:(num_of_true_in_mask, d_model)
    输出：(num_of_true_in_mask, d_model)

    本质就是FC层：d_model->4*d_model->d_model的过程
    '''
    def __init__(self, d_model:int, dropout:float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    '''
    x: (batch_size, seq_len, d_model)
    return: (batch_size, seq_len, expert_num)  表示：针对每个token，每个专家的的权重，0表示不参与
    '''
    def __int__(self, d_model:int, expert_num:int, topk_num):
        self.topk_num = topk_num
        self.topk_linear = nn.Linear(d_model, expert_num)

        self.noisy_linear = nn.Linear(d_model, expert_num)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model) -> logits: (batch_size, seq_len, expert_num)
        logits = self.topk_linear(x)

        '''
        噪音添加，到最终的噪声张量，其每个元素服从均值：0，标准差：softplus(noisy_logits)
        1. 噪音强度可学习，当过度依赖某些专家时，该专家的方差可能被学大，从而降低被选中的概率
        2. Softplus 而不是 ReLU 或直接取绝对值，因为Softplus 是光滑的，处处可导，有利于梯度优化。
        '''
        # 添加一些noisy，noisy_logits：(batch_size, seq_len, expert_num)
        noisy_logits = self.noisy_linear(x)
        # noise：(batch_size, seq_len, expert_num)
        noise = torch.randn_like(logits) * torch.nn.functional.softplus(noisy_logits)
        # noisy_logits：(batch_size, seq_len, expert_num)
        noisy_logits += noise

        '''
        topk：（k_num，dim）
        return：
            indice：表示指定维度上的index
            val: 符合top_k的tensor子集
            indice和val是一一对应关系的。
        注意：topk本身从数学上来说是不可导的，因为topk是对index进行的选择或操作，被可导。但topk实现了反向传播，
        其反向传播的功能为：针对被选中的元素，导数可以继续传播，对于没被选中的元素，导数不继续传播，导数为0！
        '''
        #  logits: (batch_size, seq_len, expert_num) -> topk_logits & topk_indice: (batch_size, seq_len, topk_num)
        topk_logits, topk_indice = logits.topk(self.topk_num, -1)
        # zeros: (batch_size, seq_len, expert_num)
        neg_inf = torch.full_like(logits, float("-inf"))

        '''
        scatter:
        dim = 0
        self[ indice[i][j][k]]  [j][k] = src[i]  [j][k]
        
        dim = 1
        tensor[i]  [indice[i][j][k]]  [k] = src[i]  [j]  [k]
        
        1. 指定一个维度
        2. 在指定维度上，维度的映射pattern就是indice矩阵
        3. 其余维度是不变的。
        '''
        # 将topk，按照topK的位置，将值填写在zeros矩阵里，其余的位置不填，空着，保持-inf的值
        # sparse_logits：(batch_size, seq_len, expert_num)
        sparse_logits = neg_inf.scatter(-1, topk_indice, topk_logits)

        # return (batch_size, seq_len, expert_num)
        return torch.softmax(sparse_logits, -1), topk_indice


class SparseMoe(nn.Module):
    def __init__(self, expert_num:int, topk_num:int, d_model:int, dropout:float):
        super().__init__()
        self.router = NoisyTopkRouter(d_model, expert_num, topk_num)
        self.experts = nn.ModuleList([Expert(d_model, dropout) for _ in range(expert_num)])
        self.topk_num = topk_num
    def forward(self, x):
        # 1. 获取路由结果
        # x: (batch_size, seq_len, d_model) -> gating_output: (batch_size, seq_len, expert_num)
        gating_output, topk_indices = self.router(x)

        final_output = torch.zeros_like(x)

        # 2. 展平batch数据
        # 输入和gate的batch都展平了，维度降低为2维
        # x: (batch_size, seq_len, d_model) -> flat_x: (batch_size * seq_len, d_model)
        flat_x = x.view(-1, x.size(-1))
        # (batch_size, seq_len, expert_num) -> (batch_size * seq_len, expert_num)
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 3. 遍历每一个expert
        for expert_ind, expert in enumerate(self.experts):

            # 4. 通过路由结果，获取当前 expert_mask
            # 每个token中，dim=-1这一行，只要有一个indice为i，说明至少有一个token用到了这个expert，，则为true，
            # topk_indices: (batch_size, seq_len, topk_num) -> expert_mask: (batch_size, seq_len)
            expert_mask = (topk_indices ==  expert_ind).any(dim=-1).view(-1)
            # expert_mask: (batch_size, seq_len) -> (batch_size * seq_len)
            flat_expert_mask = expert_mask.view(-1)

            # 只要这个专家在所有的batch的所有token中，至少一个被选中了一次
            if flat_expert_mask.any():

                # 5. 截取分发给当前expert的数据，长度从batch_size * seq_len -> num_of_true_in_mask
                '''
                # 以数组作为索引，又叫花式索引：等价于（但效率远高于）列表推导
                    [flat_x[i] for i in range(N) if flat_expert_mask[i]]
                因此要求，花式索引mask和data首维度要一致
                '''

                # flat_x: (batch_size * seq_len, d_model) 与 flat_expert_mask:(batch_size * seq_len)
                #                           -> expert_input：(num_of_true_in_mask, d_model)
                expert_input = flat_x[flat_expert_mask]

                # 6. 只将当前expert关注的数据输入给当前expert，其余的数据expert不关心
                # 维度不变：(num_of_true_in_mask, d_model) -> (num_of_true_in_mask, d_model)
                expert_output = expert(expert_input)

                '''
                组合索引：
                 flat_gating_output[flat_expert_mask, expert_ind]
                 1. rslt = flat_gating_output[flat_expert_mask]
                 2. 对rslt第二个维度中的expert_ind切片进行截取，其余的不要
                快速提取这些 token 在该专家上的分数
                '''

                # 7. 获取权重
                # flat_gating_output: (batch_size * seq_len, expert_num)  每个token下，各专家权重
                # flat_expert_mask: (batch_size * seq_len)
                # 先得到 -> (num_of_true_in_mask, expert_num)
                # 再取第i列 -> (num_of_true_in_mask,)
                gating_scores = flat_gating_output[flat_expert_mask, expert_ind].unsqueeze(1)

                # 8. 获取加权后的结果
                # (num_of_true_in_mask, d_model) * (num_of_true_in_mask,) -> (num_of_true_in_mask, d_model)
                weighted_output = expert_output * gating_scores

                # 9. 权重叠加
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output