import pathlib

import torch
import torch.nn as nn
import torch.utils.data
from config import get_config, get_weights_file_path,latest_weights_file_path
import datasets
# 如下为自定的库
import dataset
import model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    '''
    输入是配置信息：src语言，tgt语言，seq_len, batch_size等
    :param config：src语言，tgt语言，seq_len, batch_size等
    :return: src和tgt的tokenizer, src和tgt的dataloader
    '''
    # 将存储在磁盘上的原始文件（如 .csv、.json、.txt、.parquet）读取并解析为内存中的表格结构。
    ds_raw = datasets.load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # build tokenizer
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw,(train_ds_size, val_ds_size))

    train_ds = dataset.BilingualDataset(train_ds_raw, src_tokenizer, tgt_tokenizer,
                                        config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = dataset.BilingualDataset(val_ds_raw, src_tokenizer, tgt_tokenizer,
                                      config["lang_src"], config["lang_tgt"], config["seq_len"])

    '''
    dataset和dataloader的配合：
    dataset：提供单数据的索引方式（实现getitem， iter等索引接口）
    dataloader：负责每次提供提供一个batch的数据，直到结束，同时提供索引算法shuffle等。
    因此dataloader每次返回的值都是dataset的getitem返回值，前面加上一个batch维度
    '''
    train_dataloader = torch.utils.data.DataLoader(train_ds,batch_size=config["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)

    # 查看seq len
    max_src_len = 0
    max_tgt_len = 0
    for item in ds_raw:
        src_ids = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tgt_tokenizer.encode(item["translation"][config["lang_tgt"]]).ids
        max_src_len = max(len(src_ids), max_src_len)
        max_tgt_len = max(len(tgt_ids), max_tgt_len)

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

    '''
    针对一个model 再用torch.save后，通常save了如下内容：
    1. 模型参数（model.state_dict()）
    内容：一个有序字典（OrderedDict），包含模型中所有可学习参数（如权重、偏置）的张量值。
    作用：
    推理/评估：加载后可以直接用于模型的前向传播，无需重新训练。
    继续训练：作为训练的起点，避免从头开始。

    2. 优化器状态（optimizer.state_dict()）
    内容：优化器的内部状态，包括：
    参数组的超参数（如学习率、动量系数、权重衰减等）。
    每个参数对应的缓存变量（如 Adam 的一阶矩、二阶矩，SGD 的动量缓冲区）。
    作用：
    继续训练：恢复优化器的内部状态，确保后续更新与中断前的行为一致。例如，Adam 需要之前累积的梯度动量，否则重新开始会导致训练波动甚至不收敛。
    学习率调度：有些调度器依赖优化器状态来调整学习率。

    3. 当前 epoch 数
    内容：已经完成训练的 epoch 数量（通常从 0 开始计数）。
    作用：
    继续训练：从该 epoch 的下一个 epoch 开始，便于训练日志的连续性和学习率调度器（如 StepLR）的正确恢复。
    断点续训：如果训练意外中断，可以从中断点继续，避免重复工作。

    4. 全局步数（global_step）
    内容：已经执行的总迭代次数（一个 batch 为一步）。
    作用：
    日志与监控：恢复 TensorBoard 等工具的步数曲线，使图表连续。
    学习率调度：某些调度器（如 CosineAnnealingWarmRestarts）依赖全局步数进行更新。
    混合精度训练：如果使用 GradScaler，可能需要保存其状态（通常单独保存）。

    5. 损失值（当前 epoch 的平均损失或最佳损失）
    内容：最近一次验证损失或训练损失（可选）。
    作用：
    早停与模型选择：如果保存的是最佳模型（基于验证损失），可以记录该损失值以便比较。
    监控训练趋势：恢复训练时了解当前性能水平。

    6. 随机数生成器状态（可选）
    内容：PyTorch、NumPy、Python 内置 random 模块的随机状态。
    作用：
    保证可重复性：如果实验需要完全复现随机过程（如数据打乱、dropout 模式），保存这些状态可以在恢复训练时保持相同的随机序列。但一般不常用，因为通常只需保证每次运行固定种子即可
    '''

    # 1. device的选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"we are using {device} to train")

    pathlib.Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # 2. 构造dataset
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)

    # 3. 构造model
    tmodel = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())

    # 4.tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(config["experiment_name"])

    # 5.optimizer
    optimizer = torch.optim.Adam(tmodel.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' \
        else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"preloading model {model_filename}")

        state = torch.load(model_filename) # 加载checkpoint的文件（由torch.save()而来）
        initial_epoch = state['epoch'] + 1 # 恢复epoch，进行下一轮计算
        optimizer.load_state_dict(state["optimizer_state_dict"])  # 恢复优化器状态
        global_step = state["global_step"]  # 恢复迭代step

    # 6. loss function的定义(loss_fn和卷积层一样，也是一种module)
    # ignore_index 表示对某些token的输出，通过掩码的形式置为0，从而在计算loss的时候，这部分不计入loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    # 7. 迭代训练
    for epoch in range(initial_epoch, config["num_epochs"]):
        # 1). 设置模型为train模式
        # model.train() 是 PyTorch 中用于将模型设置为训练模式的方法。它会递归地将模型及其所有子模块的 training 标志设为 True，
        # 从而影响某些特定层（如 Dropout、BatchNorm 等）在 forward 中的行为。这是因为model在train和eval的时候，某些层的行为有差异
        tmodel.train()
        batch_iterator = tqdm(train_dataloader, desc= f"processing epoch {epoch:02d}") # tqdm接受迭代器，输出进度条
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (B,1,1,seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, seq_len, seq_len)

            # 2). 获取输出前向传播的输出
            encoder_output = tmodel.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = tmodel.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = tmodel.project(decoder_output)  # (B, seq_len, vocab_size)

            # 3). 计算loss
            label = batch["label"].to(device) # (B, seq_len)
            '''
            proj_output的shape：(B, seq_len, vocab_size)->(B * seq_len, vocab_size)
            label的shape：(B, seq_len) -> (B * seq_len, )
            loss_fn的输入要求input维度 (N,C) 和 (N,) 其中N表示多少个结果，C表示class，既每种结果，不同class的概率分别是多少
            loss function会对每一个样本的所有class取softmax的均值，然后和label作差，然后样本差取平均。
            因此不涉及broadcast
            '''
            loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            # tqdm的显示
            batch_iterator.set_postfix({f'loss:': f'{loss.item():6.3f}'})
            # tenorboard 可视化
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # 4). 反向传播
            loss.backward()

            # 5). 更新梯度
            optimizer.step()

            # 6). 梯度清理
            optimizer.zero_grad()

            global_step += 1

        # 8. 保存模型
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        torch.save(
            {
                'epoch':epoch,
                'model_state_dict':tmodel.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'global_step':global_step
            }, model_filename
        )

if __name__ == "__main__":
    config = get_config()
    train_model(config)












