import pathlib

import torch
import torch.nn as nn
import torch.utils.data

import datasets
# 如下为自定的库
import dataset
import model
from torch.utils.tensorboard import SummaryWriter

'''
tokenizers 是可训练的，并负责将文本拆分为原语（token），并将每个原语转换为模型可接受的数字 ID。
token可以是词，例如love，id是数字：例如30，
则tokenizer可以将[hello word，EOS]转换为[31,64,0]的向量。


tokenizers 内部的层次：
Tokenizer 是总控制器，它组合了 model（分词模型）、pre_tokenizer（预分词器）、post_processor（后处理器）、normalizer（归一化器）等组件。
WordLevel 是 model 的一种具体实现，定义了如何将词序列进一步拆分成 token（或保持原样）。
WordLevelTrainer 是专门用来训练 WordLevel 模型的工具，它不直接用于推理。
Whitespace 是 pre_tokenizer 的一种，决定了文本切分的第一阶段。
'''
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
'''
功能：WordLevel 是一种分词模型，实现了最简单的词级别分词。
作用：
它维护一个词汇表，将每个完整的词（或字符串）映射到一个唯一的 ID。
对于不在词汇表中的词，通常会使用一个特殊的 [UNK] token 代替。
与 BPE 或 WordPiece 不同，它不会将词拆分成更小的子词单元，因此词汇表通常很大，且对未登录词处理能力较弱。
用途：适用于词汇表相对固定、未登录词较少的场景，或作为学习分词器的基础示例。
'''
from tokenizers.trainers import WordLevelTrainer
'''
功能：WordLevelTrainer 是专为 WordLevel 模型设计的训练器。
作用：
它负责从语料中统计词频，构建词汇表，并将训练好的词汇表设置到 WordLevel 模型中。
训练时需要指定一些参数，如词汇表大小、特殊 token 等。
与 WordLevel 的关系：WordLevel 定义模型结构（词汇表映射），而 WordLevelTrainer 提供训练算法来填充这个词汇表。
'''
from tokenizers.pre_tokenizers import Whitespace
'''
作用：
它按照空白字符（空格、制表符、换行等）将文本切分成“词”的序列，这些词将成为后续分词模型的输入。
例如，输入 "Hello, world!" 会被切分成 ["Hello,", "world!"]（注意标点符号仍附着在词上，后续可进一步处理）。
重要性：预分词是分词流程的第一步，决定了模型看到的“词单元”边界。Whitespace 是最常用的预分词器之一，其他还有 ByteLevel、Metaspace 等。
'''
from pathlib import Path


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    '''
    配置tokenizer的步骤：
    实例化tokenizer->设置pre_tokenier类型->设置trainer->用trainer训练tokener->保存tokener
    :param config: 路径
    :param ds: 数据
    :param lang: 语言名称
    :return: 一个已经训练好的tokenizer
    '''
    # 这里 用 lang 替代 tokenizer_{}.json中的{}，例如tokenizer_en.json文件。
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # 配置一个以word这个层面为level的分词器，既不继续向下分词了，同时，对于不知的词，以unk统一代替
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        # 特殊的token，EOS结尾，SOS开始，PAD用来token对其的，UNK不知道的单词
        # min_frequence指的是只有出现两次以上的单词，才加入到词汇表中。
        trainer = WordLevelTrainer(special_tokens=["[EOS]","[UNK]","[PAD]","[SOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        # 保存，分词器会一jasn的格式来保存
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer

def get_ds(config):
    '''
    输入是配置信息：src语言，tgt语言，seq_len, batch_size等
    :param config：src语言，tgt语言，seq_len, batch_size等
    :return: src和tgt的tokenizer, src和tgt的dataloader
    '''
    # 将存储在磁盘上的原始文件（如 .csv、.json、.txt、.parquet）读取并解析为内存中的表格结构。
    ds_raw = datasets.load_dataset("opus_book", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    # build tokenizer
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tgt_tokenizer = get_or_build_tokenizer((config, ds_raw, config["lang_tgt"]))

    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw,(train_ds_size, val_ds_size))

    train_ds = dataset.BilingualDataset(train_ds_raw, src_tokenizer, tgt_tokenizer,
                                        config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = dataset.BilingualDataset(val_ds_size, src_tokenizer, tgt_tokenizer,
                                      config["lang_src"], config["lang_tgt"], config["seq_len"])

    train_dataloader = torch.utils.data.DataLoader(train_ds,batch_size=config["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)

    # 查看seq len
    max_src_len = 0
    max_tgt_len = 0
    for item in ds_raw:
        max_src_len = max(src_tokenizer.token_to_id(item["translation"][config["lang_src"]],dtype=torch.int64),max_src_len)
        max_tgt_len = max(tgt_tokenizer.token_to_id(item["translation"][config["seq_len"]],dtype=torch.int64),max_tgt_len)

    print(f'max src_len is {max_src_len}')
    print(f"max tgt len is {max_src_len}")

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, vocab_src_len, vocab_tgt_len):
    return model.build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])


def train_model(config):
    '''
    pytorch中，模型训练的步骤：

    1. optimizer的选择
        optimizer = torch.optim.Adam(model.parameters())

        optimizer主要负责：管理参数和更新参数
            1）保存model的所有参数的引用
            2）通过step，利用每一个param的data和grad，进行参数更新，其更新算法不同，分成SGD，Adam等
            3）对所有param的grad进行清空
    2. 模型前向传播：
        output = model(input)

    3. 获取loss
        loss = loss_fn(outputs, labels)

    4. loss反向传播，autograd引擎更新所有param的grad的值
        loss.backward()

    5. optimizer更新参数
        optimizer.step()

        optimizer递归的将所有param.data与param.grad进行作用，更新param.data

    6. 清理所有grad
        optimizer.zero_grad()

        optimizer递归的将所有param.grad原地复制为0
    '''

    # 1. device的选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"we are using {device} to train")

    pathlib.Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # 2. 构造dataset
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)

    # 3. 构造model
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())

    # 4.tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(config["experiment_name"])

    # 5.optimizer
    optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=config["lr"], eps=1e-9)



