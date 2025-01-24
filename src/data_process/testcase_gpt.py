import math
import inspect
from dataclasses import dataclass
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
import csv
from data_process.pre_data import csv2txt
from paths import Paths
from torch.nn import functional as F
from tqdm import tqdm

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
        print(p)

###################################################################################################
def main(program, version):
    version_dir = Paths.get_d4j_version_dir(program, version)
    print("load coverage information matrix")
    if not (version_dir / "matrix.txt").exists():
        csv2txt(version_dir)
    f1 = open(version_dir / "matrix.txt", 'r', encoding='utf-8')
    f2 = open(version_dir / "error.txt", 'r', encoding='utf-8')
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

    if inputs_pre== 0:
        print("skip！", nums[version])
        sys.exit()

    numlist = list(set(j.item() for i in inputs_pre for j in i))
    #print(numlist)
    word_dict = {}
    for i, w in enumerate(numlist):
        word_dict[w] = i
    number_dict = {i: w for i, w in enumerate(word_dict)}
    # print(number_dict)
    token_list = list()
    for hang in inputs_pre:
        arr = []
        for s in hang:
            s = s.item()
            arr.append(word_dict[s])
        token_list.append(arr)
    print(np.array(token_list).shape)
    inputs_pre = torch.tensor(np.array(token_list), device="cuda", dtype=torch.int64)

    #inputs_pre = torch.tensor(np.array(inputs_pre).astype(np.float64), device="cuda", dtype=torch.float64)
    def find_keys_by_value(dictionary, value):

        return [key for key, val in dictionary.items() if val == value]
#################################################################
    class GPTConfig:
        block_size: int = pd.read_csv(version_dir / "matrix.csv").shape[1] - 1
        vocab_size: int = len(number_dict) + 10000 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 3
        n_head: int = 3
        n_embd: int = 12
        dropout: float = 0.2
        bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    batchsize=2
    class LayerNorm(nn.Module):
        """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        def forward(self, input):
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):

        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            # key, query, value projections for all heads, but in a batch
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            # output projection
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            # regularization
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout
            # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

        def forward(self, x):
            B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                     dropout_p=self.dropout if self.training else 0,
                                                                     is_causal=True)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y

    class MLP(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x)
            return x

    class Block(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):

        def __init__(self, config):
            super().__init__()
            assert config.vocab_size is not None
            assert config.block_size is not None
            self.config = config

            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

            # init all weights
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

            # report number of parameters
            print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        def get_num_params(self, non_embedding=True):
            """
            Return the number of parameters in the model.
            For non-embedding count (default), the position embeddings get subtracted.
            The token embeddings would too, except due to the parameter sharing these
            params are actually used as weights in the final layer, so we include them.
            """
            n_params = sum(p.numel() for p in self.parameters())
            if non_embedding:
                n_params -= self.transformer.wpe.weight.numel()
            return n_params

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

            # forward the GPT model itself
            tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)

            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                #print(logits.shape,type(logits),"logits")
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
                loss = None

            return logits, loss

        def crop_block_size(self, block_size):
            # model surgery to decrease the block size if necessary
            # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
            # but want to use a smaller block size for some smaller, simpler model
            assert block_size <= self.config.block_size
            self.config.block_size = block_size
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
            for block in self.transformer.h:
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

        def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")

            return optimizer

        def estimate_mfu(self, fwdbwd_per_iter, dt):
            """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
            # first estimate the number of flops we do per iteration.
            # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
            N = self.get_num_params()
            cfg = self.config
            L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
            flops_per_token = 6 * N + 12 * L * H * Q * T
            flops_per_fwdbwd = flops_per_token * T
            flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
            # express our flops throughput as ratio of A100 bfloat16 peak flops
            flops_achieved = flops_per_iter * (1.0 / dt)  # per second
            flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
            mfu = flops_achieved / flops_promised
            return mfu

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=0.1, top_k=None):
            """
            Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
            the sequence max_new_tokens times, feeding the predictions back into the model each time.
            Most likely you'll want to make sure to be in model.eval() mode of operation for this.
            """
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                # forward the model to get the logits for the index in the sequence
                logits, _ = self(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

            return idx
    print(gnum)
    model=GPT(GPTConfig).to('cuda')
    eposes=100
    optimizer=model.configure_optimizers(1e-5 ,0.001,(0.9, 0.999),'cuda')
    for epose in tqdm(range(eposes),desc='训练进度',position=0):
        prelenth = 0
        for lenth in range(1,int(len(inputs_pre)/batchsize)+1):
            logits, loss=model(inputs_pre[prelenth*batchsize:lenth*batchsize,0:-1],inputs_pre[prelenth*batchsize:lenth*batchsize,1:])
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            prelenth=lenth
        if epose%2000==0:
            print(epose,loss)
    #print('trian')
    print(gnum)

    x=torch.empty((0,pd.read_csv(version_dir / "matrix.csv").shape[1]-1),device='cuda')
    # print(x)
    # print(x.shape)
    for i in tqdm(range(int(gnum/batchsize)),desc='生成批次进度',position=0):
        indices1 = torch.randint(0, len(inputs_pre), (batchsize,))
        log_var = inputs_pre[indices1,int(pd.read_csv(version_dir / "matrix.csv").shape[1]*0.95)].reshape(batchsize,-1)
        x1=model.generate(log_var,pd.read_csv(version_dir / "matrix.csv").shape[1]-2)
        x=torch.cat((x,x1),0)
    matrix=np.array(x.to('cpu'))
    print('matix',matrix.shape)
    # outputlast=[]
    # for i,p in enumerate(matrix):
    #     output=[]
    #     for index,value in enumerate(p):
    #         print(find_keys_by_value(number_dict,value))
    #         print(matrix[i][index])
    #         output.append(find_keys_by_value(number_dict,value))
    #         #print(output)
    #     outputlast.append(output)
    #     print(outputlast)
    # outputlast=np.array(matrix)
    path_w = version_dir / "matrix_gpt.csv"
    print(path_w)
    save_matrix_as_csv(matrix, version_dir, path_w, minority_label)
    print("generate complete")
if __name__ == "__main__":
    main('clac', '21-mr')