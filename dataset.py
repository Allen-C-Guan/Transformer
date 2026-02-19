import torch
from torch.utils.data import dataset
from tokenizers import Tokenizer
'''
在transformer的decoder中，训练采用的是teacher forcing的策略。
什么是teacher forcing？
teacher forcing指的是，不管你本次预测的token是啥，我们都会忽略掉，在下一轮的预测中，都会采用正确的预测值来预测下一个token，避免错误累计！

例如：
如果正确答案是：A,B,C,D，E

但如果
本次的输入：sos,A,B
本次的输出是: D

那么在下一次的预测中：
输入是：sos,A,B,C （而不是 SOS A B D！我们要假设上一步预测正确的情况下，来继续预测！）
输出是：D

因此对于label和decoder_input之间就会存在左移对其的规律：
假设目标序列是 ["I", "love", "Python", "<EOS>"]。
encoder_input: 解码器输入         ["<SOS>","I", "love", "Python", "<EOS>"] # 这里没有时间概念，encoder的输入永远是全量输入
                                                                         # EOS也是要学习的一部分。

如下两个序列是有时间概念的，确定一个index，既确定了一个时刻，则该时刻的输入对应的输出也随之而确定。
decoder_input：解码器输入：        ["<SOS>", "I", "love", "Python"]   # 没有必要将eos作为输入，从而让decoder去预测。
label：解码器预测的ground truth为： ["I", "love", "Python", "<EOS>"]

既：
输入 <SOS> → 预测 "I"
输入 "I" → 预测 "love"
输入 "love" → 预测 "Python"
输入 "Python" → 预测 <EOS>
'''

class BilingualDataset(torch.utils.data.Dataset):
    '''
    Dataset要实现的功能是什么？
    输入：
    1. 原生的data（英文，中文等）
    2. 训练好的tokenizer转换器
    3. 序列长度
    输出：
    实现序列索引时，将row data中的每一个item（tgt句子，src句子）转换为src token tensor，tgt_input_token tensor, label_token tensor等(ground truth)
    注意：token中要包含sos， padding，eos等
    '''
    def __init__(self, ds, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        # ds的数据结构是：第一层是id和translation为key的map，translation中又是en和ch的map
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 先把需要填充的special token计算好
        # 一般情况下，如果编码器和解码器共享词汇表，那么 <pad>、<bos>、<eos> 的 token ID 是一致的。
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        get_rslt_pair = self.ds[index]
        src_lang_text = get_rslt_pair["translation"][self.src_lang]
        tgt_lang_text = get_rslt_pair["translation"][self.tgt_lang]

        # encode(*)返回的是encoding， encoding有ids接口，返回所有id对应的列表 type = List[int]
        enc_input_tokens = self.tokenizer_src.encode(src_lang_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_lang_text).ids

        enc_pad_len = self.seq_len - len(enc_input_tokens) - 2
        dec_pad_len = self.seq_len - len(dec_input_tokens) - 1

        if enc_pad_len < 0 or dec_pad_len < 0:
            raise ValueError("sentence length error")
        # encoder input: [[sos], [I], [love], [you], [EOS], [pad], [pad]...]
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_pad_len, dtype = torch.int64),
            ],
            dim=0)

        # decoder input : [[sos], [I], [love], [you], [pad], [pad]...]
        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64)
            ]
        )

        # label(ground truth) [[I], [love], [you], [EOS], [pad], [pad]...]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64)
            ]
        )
        # 前面的类型都是tensor.size()，seq_len是int，需要让tensor返回某个维度的size，然后再比较
        assert enc_input.size(0) == dec_input.size(0) == label.size(0) ==  self.seq_len
        return {
            "encoder_input":enc_input,
            "decoder_input":dec_input,
            "label":label,
            # 将所有等于padding的位置，全部置为false，在将boolen转换为int类型
            "encoder_mask":(enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)的矩阵
            # (1, 1, seq_len) & (1, seq_len, seq_len) -> (1, seq_len, seq_len) 只保留下三角（包括对角），上三角为false
            "decoder_mask":(dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len),
            "src_text": src_lang_text,
            "tgt_text": tgt_lang_text
        }

"""
diagonal = 0时表示主对角线。
1 表示对角线上移一步
-1 表示下移一步
triu会将对角线及以上的内容保留，对角线以下的置为0
"""
def causal_mask(size):
    mask = torch.ones(1, size, size)
    mask = torch.triu(mask, diagonal=1).int() # 上三角，且不包含对角线上，全部为1，其余为0
    """
    diagonal = 0时表示主对角线。
    1 表示对角线上移一步
    -1 表示下移一步
    triu会将对角线及以上的内容保留，对角线以下的置为0
    """
    mask = mask == 0  # 下三角，包括对角线，全部为true，其余为false
    return mask

