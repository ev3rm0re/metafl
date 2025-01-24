## from https://github.com/graykode/nlp-tutorial/tree/master/5-2.BERT
import pandas as pd
import math
import re
from random import *
import torch.optim as optim
import csv
import sys
import torch.autograd
import torch.nn as nn
import torch
import numpy as np
from data_process.pre_data import csv2txt
from paths import Paths
def main(program, version):
    def find_keys_by_value(dictionary, value):
        return [key for key, val in dictionary.items() if val == value]
    def save_matrix_as_csv(data, version_dir, path_w, label):
        df = pd.read_csv(version_dir / "matrix.csv")
        header_list = list(df.columns)[0:-1]
        header_list.append('error')
        error_list = []
        # for i in len(data[0]):
        # data=np.concatenate([data,label.reshape(-1,1)],axis=1)
        # print(data.head(5))
        with open(path_w, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            data = np.insert(data, len(data[0]), values=label, axis=1)
            # print(data.shape,8)
            p = 0
            for d in data:
                p = p + 1
                writer.writerow(d)
            # print(p)
    def random_select_rows(two_d_list, num):
        selected_rows = []
        for _ in range(num):
            # 随机选择一个索引
            index = randint(0, len(two_d_list) - 1)
            # 将选中的行添加到结果列表中
            selected_rows.append(two_d_list[index])
        return selected_rows
    def make_batch():
        batch = []
        positive = negative = 0  ## 为了记录NSP任务中的正样本和负样本的个数，比例最好是在一个batch中接近1：1

        # while positive != batch_size / 2 or negative != batch_size / 2:
        while positive +negative<batch_size:
            tokens_a_index, tokens_b_index = randrange(minority_num), randrange(
                minority_num)  # 比如tokens_a_index=3，tokens_b_index=1；从整个样本中抽取对应的样本；
            tokens_a, tokens_b = token_list[tokens_a_index], token_list[
                tokens_b_index]  ## 根据索引获取对应样本：tokens_a=[5, 23, 26, 20, 9, 13, 18] tokens_b=[27, 11, 23, 8, 17, 28, 12, 22, 16, 25]
            input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict[
                                                                                                 '[SEP]']]  ## 加上特殊符号，CLS符号是1，sep符号是2：[1, 5, 23, 26, 20, 9, 13, 18, 2, 27, 11, 23, 8, 17, 28, 12, 22, 16, 25, 2]
            segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (
                    len(tokens_b) + 1)  ##分割句子符号：[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            # MASK LM
            n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # n_pred=3；整个句子的15%的字符可以被mask掉，这里取和max_pred中的最小值，确保每次计算损失的时候没有那么多字符以及信息充足，有15%做控制就够了；其实可以不用加这个，单个句子少了，就要加上足够的训练样本
            # print('input_ids:',len(input_ids) * 0.15)
            # print('n_pred:',n_pred)
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                              if token != word_dict['[CLS]'] and token != word_dict[
                                  '[SEP]']]  ## cand_maked_pos=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]；整个句子input_ids中可以被mask的符号必须是非cls和sep符号的，要不然没意义
            # print(np.array(cand_maked_pos).shape)
            shuffle(cand_maked_pos)  ## 打乱顺序：cand_maked_pos=[6, 5, 17, 3, 1, 13, 16, 10, 12, 2, 9, 7, 11, 18, 4, 14, 15]  其实取mask对应的位置有很多方法，这里只是一种使用shuffle的方式

            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[
                       :n_pred]:  ## 取其中的三个；masked_pos=[6, 5, 17] 注意这里对应的是position信息；masked_tokens=[13, 9, 16] 注意这里是被mask的元素之前对应的原始单字数字；
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                if random() < 0.8:  # 80%
                    input_ids[pos] = word_dict['[MASK]']  # make mask
                elif random() < 0.5:  # 10%
                    index = randint(0, vocab_size - 1)  # random index in vocabulary
                    input_ids[pos] = word_dict[number_dict[index]]  # replace

            # Zero Paddings
            n_pad = maxlen - len(input_ids)  ##maxlen=30；n_pad=10
            input_ids.extend([0] * n_pad)  # 在input_ids后面补零
            segment_ids.extend([0] * n_pad)  # 在segment_ids 后面补零；这里有一个问题，0和之前的重了，这里主要是为了区分不同的句子，所以无所谓啊；他其实是另一种维度的位置信息；

            # Zero Padding (100% - 15%) tokens 是为了计算一个batch中句子的mlm损失的时候可以组成一个有效矩阵放进去；不然第一个句子预测5个字符，第二句子预测7个字符，第三个句子预测8个字符，组不成一个有效的矩阵；
            ## 这里非常重要，为什么是对masked_tokens是补零，而不是补其他的字符？？？？我补1可不可以？？
            if max_pred > n_pred:
                n_pad = max_pred - n_pred
                masked_tokens.extend([
                                         0] * n_pad)  ##  masked_tokens= [13, 9, 16, 0, 0] masked_tokens 对应的是被mask的元素的原始真实标签是啥，也就是groundtruth
                masked_pos.extend([0] * n_pad)  ## masked_pos= [6, 5, 17，0，0] masked_pos是记录哪些位置被mask了

            # if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            if tokens_a_index + 1 == tokens_b_index:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
                positive += 1
            #elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            elif tokens_a_index + 1 != tokens_b_index :
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # NotNext
                negative += 1
            # print("positive:",positive)
            # print("negative:", negative)
        return batch

    # Proprecessing Finished

    def get_attn_pad_mask(seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

    def gelu(x):
        "Implementation of the gelu activation function by Hugging Face"
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    # sample IsNext and NotNext to be same in small batch size
    class Embedding(nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.tok_embed = nn.Embedding(vocab_size, d_model,device=device)  # token embedding
            self.pos_embed = nn.Embedding(maxlen, d_model,device=device)  # position embedding
            self.seg_embed = nn.Embedding(n_segments, d_model,device=device)  # segment(token type) embedding
            self.norm = nn.LayerNorm(d_model,device=device)

        def forward(self, x, seg):

            seq_len = x.size(1)
            pos = torch.arange(seq_len, dtype=torch.long,device=device)
            pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
            # print(x.shape,111)
            # print(pos.shape,222)
            # print(seg.shape,333)
            # print(seg[0])
            # print("Input ids:", input_ids.shape)
            # print("Embedding weight shape:", self.pos_embed.weight.shape)
            # print('pos:',pos.shape)
            # print(self.pos_embed(pos))
            # print(self.seg_embed(seg)) 
            # print(self.tok_embed(x))
            embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
            return self.norm(embedding)

    class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask):
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
                d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V)
            return context, attn

    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_k * n_heads,device=device)
            self.W_K = nn.Linear(d_model, d_k * n_heads,device=device)
            self.W_V = nn.Linear(d_model, d_v * n_heads,device=device)

        def forward(self, Q, K, V, attn_mask):
            # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
            residual, batch_size = Q, Q.size(0)
            # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            # print(n_heads)
            # print(d_k)
            q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
            k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
            v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,1)  # attn_mask : [batch_size x n_heads x len_q x len_k]
            # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
            context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1,n_heads * d_v)  # context: [batch_size x len_q x n_heads * d_v]
            output = nn.Linear(n_heads * d_v, d_model,device=device)(context)
            return nn.LayerNorm(d_model,device=device)(output + residual), attn  # output: [batch_size x len_q x d_model]

    class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
            super(PoswiseFeedForwardNet, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
            return self.fc2(gelu(self.fc1(x)))

    class EncoderLayer(nn.Module):
        def __init__(self):
            super(EncoderLayer, self).__init__()
            self.enc_self_attn = MultiHeadAttention()
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, enc_inputs, enc_self_attn_mask):
            # print(enc_inputs.shape,2024)
            # print(enc_self_attn_mask.shape,2003)
            enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                   enc_self_attn_mask)  # enc_inputs to same Q,K,V
            enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
            return enc_outputs, attn

    ## 1. BERT模型整体架构
    class BERT(nn.Module):
        def __init__(self):
            super(BERT, self).__init__()
            self.embedding = Embedding()## 词向量层，构建词表矩阵
            self.layers = nn.ModuleList([EncoderLayer()for _ in range(n_layers)])  ## 把N个encoder堆叠起来，具体encoder实现一会看
            self.fc = nn.Linear(d_model, d_model)  ## 前馈神经网络-cls
            self.activ1 = nn.Tanh()  ## 激活函数-cls
            self.linear = nn.Linear(d_model, d_model)  # -mlm
            self.activ2 = gelu  ## 激活函数--mlm
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, 2)  ## cls 这是一个分类层，维度是从d_model到2，对应我们架构图中就是这种：
            # decoder is shared with embedding layer
            embed_weight = self.embedding.tok_embed.weight
            n_vocab, n_dim = embed_weight.size()
            self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
            self.decoder.weight = embed_weight
            self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        def forward(self, input_ids, segment_ids, masked_pos):
            output = self.embedding(input_ids, segment_ids)  ## 生成input_ids对应的embdding；和segment_ids对应的embedding
            enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

            for layer in self.layers:
                output, enc_self_attn = layer(output, enc_self_attn_mask)
            # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
            # it will be decided by first token(CLS)
            h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
            logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

            masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(
                -1))  # [batch_size, max_pred, d_model]  其中一个 masked_pos= [6, 5, 17，0，0]
            # get masked position from final output of transformer.
            h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
            h_masked = self.norm(self.activ2(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, n_vocab]

            return logits_lm, logits_clsf
    version_dir = Paths.get_d4j_version_dir(program, version)
    print("load coverage information matrix")
    if not (version_dir / "matrix.txt").exists():
        csv2txt(version_dir)
    f1 = open(version_dir / "matrix.txt", 'r', encoding='utf-8')
    f2 = open(version_dir / "error.txt", 'r', encoding='utf-8')
    """
    set parameters
    """
    print("set parameters")

    first_ele = True
    for data in f1.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_x = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_x = np.c_[matrix_x, nums]
    f1.close()
    print(matrix_x.shape)

    first_ele = True
    for data in f2.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_y = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_y = np.c_[matrix_y, nums]
    f2.close()
    print("matirx_y:")
    print(matrix_y.shape)

    matrix_x = matrix_x.transpose()
    matrix_y = matrix_y.transpose()

    TESTNUM_TOTAL = len(matrix_y)
    inputs_pre = []
    fail_num = np.sum(matrix_y == 1)
    pass_num = np.sum(matrix_y == 0)
    assert fail_num != pass_num
    if fail_num < pass_num:
        minority_label = 1
        minority_num = fail_num
        gnum = pass_num - fail_num
    else:
        minority_label = 0
        minority_num = pass_num
        gnum = fail_num - pass_num
    for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == minority_label:
            inputs_pre.append(matrix_x[testcase_num])
            # inputs_pre.append(matrix_y[testcase_num])
    # print(np.array(inputs_pre).shape)
    inputs = torch.FloatTensor(np.array(inputs_pre))
    # print(inputs)
    # print(inputs.shape)
    INUNITS = len(inputs_pre[0])
    TESTNUM = minority_num
    numlist=list(set(j.item() for i in inputs for j in i ))
    #print(numlist,1)
    if inputs_pre == 0:
        print("skip！", nums[version])
        sys.exit()
        # BERT Parameters
    csvlen=pd.read_csv(version_dir / "matrix.csv").shape[1]-1
    maxlen = csvlen*2+3  # 句子的最大长度 cover住95% 不要看平均数 或者99%  直接取最大可以吗？当然也可以，看你自己
    batch_size = 5# 每一组有多少个句子一起送进去模型
    max_pred = 500  # max tokens of prediction
    n_layers = 4  # number of Encoder of Encoder Layer
    n_heads = 2  # number of heads in Multi-Head Attention
    d_model = 2 # Embedding Size
    d_ff = 8  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    device = torch.device('cuda')

    #sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
    #print(sentences)
    #word_list = list(set(" ".join(sentences).split()))
    #print(word_list)
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

    #word_dict={}
    for i, w in enumerate(numlist):
        word_dict[w] = i + 4

    number_dict = {i: w for i, w in enumerate(word_dict)}
    # print(word_dict)
    # print('find_keys_by_value',find_keys_by_value(word_dict, 4))

    vocab_size = len(word_dict)
    #print(vocab_size)
    token_list = list()
    for hang in inputs:
        arr=[]
        for s in hang:
            s=s.item()
            arr.append(word_dict[s])
        token_list.append(arr)
    # for sentence in sentences:
    #     arr = [word_dict[s] for s in sentence.split()]
    #     token_list.append(arr)
    #print(token_list)
    batch = make_batch()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))
    input_ids = input_ids.to(device)
    segment_ids = segment_ids.to(device)
    masked_tokens = masked_tokens.to(device)
    masked_pos = masked_pos.to(device)
    isNext = isNext.to(device)
    #print(type(input_ids),type(segment_ids),type(masked_pos))

    model = BERT().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids, segment_ids,masked_pos)  ## logits_lm 【6，5，29】 bs*max_pred*voca  logits_clsf:[6*2]
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM ;masked_tokens [6,5]
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext)  # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Predict mask tokens ans isNext
    # print('gnum:',gnum)
    token_list = random_select_rows(token_list, num=gnum)
    minority_num=len(token_list)
    # print('minority_num:',minority_num)
    with torch.no_grad():
        batch_size=gnum
        batch = make_batch()
        predicted_token_list = []
        for i in range(0,len(batch)) :
            input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[i]))
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            masked_tokens = masked_tokens.to(device)
            masked_pos = masked_pos.to(device)
            isNext = isNext.to(device)
            # print('input_ids shape:',input_ids.shape)
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
            # print('logits_lm:',logits_lm[0].shape)
            logits_lm = logits_lm.data.cpu().max(2)[1][0].data.numpy()
            # print('masked tokens list : ', [pos.item() for pos in masked_tokens[0] if pos.item() != 0])
            # print('predict masked tokens list : ', [pos for pos in logits_lm ])
            # print('predict masked tokens list shpe: ',np.array( [pos for pos in logits_lm ]).shape)
            # logits_clsf = logits_clsf.data.cpu().max(1)[1].data.numpy()[0]
            # print('isNext : ', True if isNext else False)
            # print('predict isNext : ', True if logits_clsf else False)
            # print('input_ids:',input_ids.shape)
            for tokens,token_indexs,predicted_id in zip(list(np.array(input_ids.cpu())),masked_pos,logits_lm,):
                for token_index ,tokens_oindex in zip(token_indexs,range(0,len(tokens)) ):
                    token_index=int(token_index.cpu().numpy())
                    #tokens_oindex=int(tokens_oindex.cpu().numpy())
                    # print('token_index:',token_index)
                    # print('tokens:', tokens)
                    # print('predicted_id:',predicted_id)
                    # print('find_keys_by_value:',find_keys_by_value(word_dict,predicted_id))
                    #tokens[token_index] =find_keys_by_value(word_dict,predicted_id)
                    if find_keys_by_value(word_dict,tokens[tokens_oindex]) is not None and len(find_keys_by_value(word_dict,tokens[tokens_oindex])) >= 1:
                        if isinstance(find_keys_by_value(word_dict,tokens[tokens_oindex])[0],str):
                            pass
                        else:
                            tokens[tokens_oindex]=find_keys_by_value(word_dict,tokens[tokens_oindex])[0]
                        if isinstance(find_keys_by_value(word_dict,predicted_id)[0],str):
                            tokens[token_index]=0
                        else:
                            tokens[token_index]=find_keys_by_value(word_dict,predicted_id)[0]
                # print('tokens:',tokens)
                predicted_token_list.append(list(tokens[1:csvlen+1]))
                predicted_token_list.append(list(tokens[csvlen+2:csvlen*2+2]))
        # print('predicted_token_list:',predicted_token_list)
        print(np.array(predicted_token_list).shape)
        matrix = np.array(predicted_token_list)
        path_w = version_dir / "matrix_bert.csv"
        print(path_w)
        save_matrix_as_csv(matrix, version_dir, path_w, minority_label)
        print("generate complete")

if __name__ == "__main__":
    main('grep','28-sf')


