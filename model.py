import math

import torch
import torch.nn as nn



class InputEmbedding(nn.Module):
    '''

    forward输入输出

    输入：(batch_size, seq_len) int64类型矩阵

    输出：(batch_size, seq_len， d_model) 增加d_model维度，且类型变为float 32矩阵
    '''
    # vocab_size 指的是词汇表的大小，既全集又多少个单词
    def __init__(self, d_model:int, vocab_size:int):  # d_model，即模型的大小，模型大小和embedding后的尺寸是一致的
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # embedding的功能就是讲indice（索引）转换为稠密向量
                                                            # 例如将“我”这个词(对应indice为3）转换为[0.1,0.3,0.2,0.5]这样的稠密的矩阵上
    # 在pytorch中，nn会自动调用forward来实现正向传播
    # Module在__call__中调用了forward，从而实现当实例被call的时候，自动执行forward

    def forward(self, x):
        # 输出维度是：(batch_size, seq_size, d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)  # 论文说，将embedding后要放大sqrt(d_model)倍
        return x


class PositionEncoding(nn.Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model) float32类型矩阵

    输出：(batch_size, seq_len， d_model)
    '''
    # position中，位置信息的vector大小，与词汇的大小保持一致。
    # seq_len：指的是句子的最大长度是多少？因为position的编码方式和矩阵最大值强相关。
    # dropout：
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # torch中广播的条件
        # 1. 对应维度相同
        # 2. 对应维度有一个为1， 则1 可以向另一个非1来广播
        # 3. 维度缺失，则补为1.（注意缺失的维度只能从左边补充！）例如[2,3,4]和[   3, 4] 左边可以补1，随后扩展成2

        pe = torch.zeros(seq_len, d_model)  # 需要一个position矩阵，
                                            # 1. 矩阵要求每个单词的位置编码不同，单词长度是seq len，因此编码方式有seq len种，所以其中一个维度就是seq_len，
                                            # 2. 另一维度为position code的向量大小，要求位置大小和embedding大小相同，因此为d_model
        # 一个(seq_len, 1)的向量
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # unsqueeze在指定维度上加一个维度
        # 一个 没有维度的向量,(d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 以2为步长产生分子

        # 这里 position * div_term的尺寸为：(seq_len, 1) * (d_model/2) -> boardcast为(seq_len, d_model/2)

        '''
        广播的本质是将纬度从左向右用for循环拆开遍历，每拆开一个for，左边维度就消失一维，直到一个维度和另一个纬度之间维度数量相同且都是2维，就可以进行矩阵运算了
        例如：
        A.shape = (2,3,4), B.shape = (4,5)则
        实际发生的是：
           C = torch.empty(2, 3, 5)
           for i in range(2):
               C[i] = torch.matmul(A[i], B)  # A[i] 形状 [3,4], B 形状 [4,5]，降到这里，两个矩阵维度都是2D，可以计算了
        '''

        # 这里pe[:, 0::2]的维度也是 seq_len, d_model/2 所以对应位置就会完成赋值！
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了让 embedding在于一个batch的数据进行计算时，能顺利的boardcast，
        # 且由于一个batch的输入尺寸应该为(batch_size, seq_size, de_model)
        # 因此针对于pe的(seq_len, de_model)，要想boardcast，需要在第一维度增加一个维度为（1，seq_len, de_model)
        pe = pe.unsqueeze(0)

        '''
        
        三者的核心区别：

        | 特性              | nn.Parameter          | register_buffer       | 普通 Attribute        |
        |-------------------|-----------------------|-----------------------|-----------------------|
        | 梯度计算          | ✅ 参与梯度计算          | ❌ 不参与梯度          | ❌ 不参与梯度          |
        | 优化器更新        | ✅ 会被优化器更新        | ❌ 不会被更新           | ❌ 不会被更新          |
        | 保存/加载         | ✅ 保存到 state_dict   | ✅ 保存到 state_dict    | ❌ 不保存              |
        | 设备管理          | ✅ 自动移动设备         | ✅ 自动移动设备          | ❌ 不会自动移动        |
        | 序列化            | ✅ 可序列化            | ✅ 可序列化             | ❌ 不保证可序列化      |
        | 类型              | torch.Tensor          | torch.Tensor          | 任意Python对象        |
        | requires_grad    | 默认 True              | 默认 False             | 不适用                |
        | 用途              | 模型权重               | 模型状态/缓存            | 配置/临时变量         |

        
        黄金规则：
        需要训练 → 用 nn.Parameter
        需要保存但不训练 → 用 register_buffer
        临时/配置 → 用普通Attribute
        不确定时 → 考虑是否需要在保存/加载时保留
        
        记住：
        state_dict()只保存Parameter和Buffer
        .to(device)只移动Parameter和Buffer
        优化器只更新Parameter
        普通Attribute适合存储配置、临时变量和运行时状态

        '''
        # position embedding 是不需要更新，但是需要保存并移动到device的，需要放到buffer里。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 输出维度是：(batch_size, seq_size, d_model)
        # 若输入的句子长度为len，则截取embedding的前x个，加到embedding code上去
        # 这里切片的操作是不需要求导的（尽管在buffer里也不会求导，但是显示的写出来）
        # 但是这里x是会被求导的
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)

        # 这里使用dropout的原因是避免过度依赖位置信息！降低泛化能力
        return self.dropout(x)

class LayNormalization(nn.Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model) float32类型矩阵

    输出：(batch_size, seq_len， d_model)
    '''
    def __init__(self, eps : float = 10 ** -6):
        super().__init__()
        # eps是防止分母为0的情况
        self.eps = eps
        # alpha 和 bias 是可以学的，这两个参数存在的原因是 对于绝对的(0，1)分布，有时候要求过于严格。
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x的维度应该是 == 输出维度是：(batch_size, seq_size, d_model(embedding_len))
        # d_model的维度求means, means的维度为(batch_size, seq_size, 1)
        mean = x.mean(-1, keepdim=True)
        '''     
        降维指的是：
        tensor([[1.5000],
        [3.5000]])
        变成 
        tensor([1.5000,
        3.5000])
        你会发现里面的1.5已经没有括号了，行维度消失了。降维就是高维向低纬度的投影，既3维被投射成了2维，2维会被压扁成为1维度。
        keepdim指的是，虽然词嵌入的维度求mean之后，由于一个维度直接变成标量，因此该维度应该被消去，但是keepdim表示，不要降低维度，
        虽然该维度的大小只有1，但是还是保持该维度，他应该是一个厚度为1的，而不是没有厚度的。
        '''

        std = torch.std(x, -1, keepdim = True)
        # scalar * {(batch_size, seq_size, d_model) - (batch_size, seq_size, 1)}/ {(batch_size, seq_size, 1) - scalar} + scalar
        x = self.alpha * (x - mean)/(std + self.eps) + self.bias
        return x # (batch_size, seq_len)


class FeedForwardBlock(nn. Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model)

    输出：(batch_size, seq_len， d_model)
    '''
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        # math: y = xA ^ T + b
        # (in_features(X维度): int, out_features(y的维度): int, bias: bool = True...)
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)  #w1和B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)  #w2和b2

    def forward(self, x):
        '''
        :param x:（batch_size, seq_size, d_model)

        :return: x: (batch_size, seq_size, d_model)
        '''
        # （batch_size, seq_size, d_model)-> ( batch_size, seq_size, d_ff) -> (batch_size, seq_size, d_model)
        '''
        pytorch中矩阵乘法的计算！
        A 的形状: [batch, m, n] = [2, 3, 4]
        B 的形状: [n, p] = [4, 5]

        运算规则：
        1. 将 A 视为 batch 个 [3, 4] 的矩阵
        2. 对每个 batch 中的矩阵与 B 进行矩阵乘法
        3. 每个 batch 的结果是 [3, 5]
        4. 最终结果是 [2, 3, 5]

        数学表示：
            对于每个 batch i:
            C[i, :, :] = A[i, :, :] @ B
        '''
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model) float32类型矩阵

    输出：(batch_size, seq_len， d_model)
    '''

    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model can not be divided by head number"
        self.d_k =  int(d_model / h)

        # 只有如下这些才是multi head Attention的parameters，其他的都是计算，不是参数！
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)

        self.w_0 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    # 在当前这个例子中，Q和K相同，且均为encode的 multi-attention后的输出
    def self_attention(query, key, value, mask, dropout:nn.Dropout):
        '''
        input: q,k,v的shape为（batch_size, head_num, seq_len, dk)， mask shape (B, 1, 1, seq_len)

        output:(batch_size, h, seq_len, dk)
        '''
        d_k = query.shape[-1]
        '''
        矩阵的乘法，只有最后两个维度是数学中矩阵乘法，前面的所有维度仅仅表示2维矩阵的组织形式而已。因此数学的矩阵乘法，只可以针对2D矩阵使用
        '''
        # query：(batch_size, h, seq_len, dk) @ key.transpose：(batch_size, h, dk, seq_len) -> (batch_size, h, seq_len, seq_len)
        # 这个矩阵表示q和k相互作用后，每两个两个token之间的关系！在乘以K后，表示综合所有token的关系，加权后的值。
        # QK相乘之后得到的attention，用mask盖住无因果关系的部分
        attention_scores = query @ key.transpose(-2,-1)/math.sqrt(d_k)
        '''
        这里mask，
        encoder调用时候，传入的维度是(batch_size, 1, 1, seq_len)
        decoder调用时， 传入的维度是(batch_size, 1， seq_len, seq_len) 既包含了因果矩阵的维度。
        '''
        if mask is not None:
            # 将mask == 0 的位置上的点，对应的改成10^-9的值，这样在softmax就可以变成0
            attention_scores = attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1) # soft max不改变尺寸，(batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, dk) -> (batch_size, h, seq_len, dk)
        return attention_scores @ value, attention_scores  # 第二项是为了可视化，看相关性而返回的



    #  # output:(batch_size, seq_len, d_model)
    def forward(self, q, k, v, mask):
        '''

        :param q: (batch_size, seq_len, d_model)
        :param k: (batch_size, seq_len, d_model)
        :param v: (batch_size, seq_len, d_model)
        :param mask: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        '''
        '''
        multi head的计算这里是采用：
        1. QKV矩阵获取：利用句子embedding，乘以w_q, w_k, w_v分别得到QKV，（这里的QKV等于 将多个head对应的w_qi, w_ki, w_vi，横向排列为大矩阵w_q, w_k, w_v，尺寸为(batch_size, seq_len, d_model), 然后矩阵相乘）
        2. 大矩阵QKV在d_model维度，拆成h个部分，每个部分长度为d_k，得到多头q，k，v、，大矩阵重新组织后，尺寸为(batch_size, head, seq_len, d_k)
        3. 以重组后的多头qkv以(batch_size, head, seq_len, d_k)互相做矩阵乘法，得到多头attention（矩阵乘法只有后两个维度是矩阵运算，所以多头之间依然的分离的，头与头之间互相不会参与计算）
                输出维度和输入一致(batch_size, head, seq_len, d_k)
        4.multi-head attention重新concat回(batch_size, seq_len, d_model)尺寸
        5.用全连接重新综合多个分离运算的head的结果，输出尺寸依然为(batch_size, seq_len, d_model)

        这里有个数学背景，既如果
        A= [[a00,a01],
            [a10,a11]]

        B = [[b00,b01],
             [b10,b11]]
        C = A * B.T
        C[0][0] = a00*b00 + a01 * b01
        但是如果把A拆开为
        A0 = [a00,a01]  A1 = [a10,a11]
        B0 = [b00,b01]  B1 = [b10,b11]
        此时C0_00 = (A0 * B0.T)[0][0] = a00*b00

        a00*b00 + a01 * b01 与 a00*b00 是不同的，前者是多个维度(0列 和 1列）互相混合一起的结果，第二列是只有第0列的数据

        因此所谓的多头，就是纵向切分后，使切分后的部分计算互不影响，随后在统一通过concat整合到一起的结果！

        而保持独立性，可以让每个头相对独立的抽象出互不相关的维度上的关联性。

        输入3D → 投影(仍3D) → 重塑4D → 并行计算 → 重塑3D → 输出
        这平衡了：
        1. 接口一致性（前后端）
        2. 计算效率（中间计算）
        3. 内存效率（权重共享）
        4. 实现简洁性（代码清晰）
        '''
        # step1： 用词语embeding计算QKV：(batch_size, seq_len, d_model)->(batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # step2： 用QKV计算attention
        # (batch_size, seq_len, d_model)->(batch_size, seq_len, head_num, d_k) -> （batch_size, head_num, seq_len, dk)
        # 最后面那个转置很重要，这意味着转置后，当前两个维度确定后，将会确定一个(seq_len, dk)，表示完整的句子在这个head上的所有特征。
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # x.shape = (batch_size, h, seq_len, d_v)
        x, self.attention_scores = MultiHeadAttentionBlock.self_attention(query, key, value, mask, self.dropout)

        # concat multi-head
        '''
        将h和dv维度的融合，实际上相当于将num_heads个d_v维向量在特征维度上拼接。既在d_v维度上堆叠长度。
        '''
        # (batch_size, h, seq_len, d_v)->(batch_size, seq_len, h, d_v)->(batch_size, seq_len, h*d_v=d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0],-1, self.d_k * self.h)
        # (batch_size, seq_len, d_model) * (batch_size, d_model, d_model) -> (batch_size, seq_len, d_model)
        x = self.w_0(x)
        return x

class ResidualConnection(nn.Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model) float32类型矩阵

    输出：(batch_size, seq_len， d_model)
    '''
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayNormalization()

    # 残差网路中，输入是x和skip layer模型，利用x和模型输出的结果y，完成了一次x+y的add和norm，
    # 针对于只有一个入参的sublayer的接口约定，更加抽象，更简洁，更普世
    def forward(self, x, sublayer):  #这里对sublayer要求只要是个一个入参为一个值的，callable的对象即可！
        x = x + self.dropout(sublayer(self.norm(x))) # 如果sublayer有入参咋弄？ 可以通过lambda封装为只有一个入参！
        return x

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))
    #  # output:(batch_size, seq_len, d_model)
    def forward(self, x, src_mask):
        # 这里通过lambda表达式，封装了self_attention_block，封装后其入参只有一个x，符合ResidualConnection.forward中sublayer的要求
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model) float32类型矩阵

    输出：(batch_size, seq_len， d_model)
    '''

    '''
    这里他只是用了layerlist作为输入，并没有在里面直接把每一个encoderblock都直接初始化出来
    '''
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock,
                 feed_forward: FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
    # output:(batch_size, seq_len, d_model)
    def forward(self,x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # 这里的输入x是目标语言，例如英译汉，这里的x是汉语
        # 在attention模型中，encoder的输出维度(batch_size, seq_len, d_model)的一个矩阵，该矩阵作为cross attention的两个输入
        ''' encoder的输出作为key和val，而decoder的中间变量作为q来查询encoder的k和v'''
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # encoder的输出是针对英文的，
        x = self.residual_connection[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    '''

    forward输入输出维度不变

    输入：(batch_size, seq_len， d_model) float32类型矩阵

    输出：(batch_size, seq_len， d_model)
    '''
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    '''

    forward输入输出维度变化

    输入：(batch_size, seq_len， d_model)

    输出：(batch_size, seq_len, vocab_size)  从d_model变成了词表所有词的概率了
    '''
    def __init__(self, d_model:int,vocab_size:int):
        super().__init__()
        # 将矩阵最后一个维度投射到另一个维度上的全连接层
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size) 而且是个概率 softmax输出
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder,
                 src_embed:InputEmbedding, tgt_embed:InputEmbedding,
                 src_pos:PositionEncoding, tgt_pos:PositionEncoding,
                 projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer =  projection_layer

    '''encoder是可以重复利用的！推理过程中可以直接使用的'''
    def encode(self, x, src_mask):
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int,
                      d_model:int = 512, N:int = 6, h:int = 8, dropout:float = 0.1, d_ff:int = 2048) -> Transformer:
    '''
    如下都是超参数：
    :param src_vocab_size: 源语言词汇表的大小，指的是全集一共有多少个单词
    :param tgt_vocab_size: 目标语言词汇表大小
    :param src_seq_len: 源句子长度
    :param tgt_seq_len: 目标句子长度
    :param d_model:
    :param N: Decoder和Encoder的个数
    :param h: head 的个数
    :param dropout:
    :param d_ff: feed forward个数
    :return: transformer实例

    transformer中，各维度的输出：

    在src部分：

    src_input: (batch_size, seq_len)

    input_embedding: (batch_size, seq_len) -> (batch_size, seq_len, d_model)

    encoder: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

    在tgt部分：
    tgt_input: (batch_size, seq_len)

    input_embedding:(batch_size, seq_len)->(batch_size, seq_len, d_model)

    decoder：input_embedding:(batch_size, seq_len, d_model)， encoder output：((batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

    project:(batch_size, seq_len， d_model) -> (batch_size, seq_len, vocab_size)
    '''
    src_embed = InputEmbedding(d_model,src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len,dropout)

    encoder_list = []
    for _ in range(N):
        attention = MultiHeadAttentionBlock(d_model, h, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(attention, ff, dropout)
        encoder_list.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_list))

    decoder_list = []
    for _ in range(N):
        self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention, cross_attention, ff, dropout)
        decoder_list.append(decoder_block)

    decoder = Decoder(nn.ModuleList(decoder_list))

    projection = ProjectionLayer(d_model, tgt_vocab_size)
    transformer =  Transformer(encoder, decoder,src_embed, tgt_embed, src_pos, tgt_pos, projection)

    # 初始化参数
    # module里调用parameter，会递归的向下调用parameter，将所有子孙阶段的param均遍历一遍。
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return transformer

