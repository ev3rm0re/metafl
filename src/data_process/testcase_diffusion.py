import csv
import sys
import pandas as pd
import torch.autograd
import torch.nn as nn
import torch
import numpy as np
from data_process.pre_data import csv2txt
from paths import Paths
import matplotlib.pyplot as plt


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机的时刻t,t的形状是torch.Size([batchsize, 1])

    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    #print(t.shape,0)
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)

    # x0的系数
    a = alphas_bar_sqrt[t]  # torch.Size([batchsize, 1])

    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]  # torch.Size([batchsize, 1])
    #print(one_minus_alphas_bar_sqrt[0],0)
    # 生成随机噪音eps
    e = torch.randn_like(x_0)  # torch.Size([batchsize, 2])

    # 构造模型的输入
    #print(aml.shape,1)
    #print(x_0.shape,2)
    #print(e.shape,3)
    x = x_0 * a + e * aml  # torch.Size([batchsize, 2])

    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))  # t.squeeze(-1)为torch.Size([batchsize])
    # output:torch.Size([batchsize, 2])
    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0],x_seq由x[T]、x[T-1]、x[T-2]|...x[0]组成"""
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    #print(x_seq[0].shape,1)
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)

    #print(x_seq[0].shape, 1)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)

    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)
def save_matrix_as_csv(data, version_dir, path_w, label):
    df = pd.read_csv(version_dir / "matrix.csv")
    header_list = list(df.columns)[0:-1]
    header_list.append('error')
    error_list = []
    #for i in len(data[0]):
    #data=np.concatenate([data,label.reshape(-1,1)],axis=1)
    #print(data.head(5))
    with open(path_w, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        data = np.insert(data, len(data[0]), values=label, axis=1)
        #print(data.shape,8)
        p=0
        for d in data:
            p=p+1
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
    """
    set parameters
    """
    print("set parameters")
    num_epoch = 1000  # 1000
    z_dimension = 100  # 100

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
        gnum =pass_num-fail_num
    else:
        minority_label = 0
        minority_num = pass_num
        gnum =fail_num -  pass_num
    for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == minority_label:
            inputs_pre.append(matrix_x[testcase_num])
            #inputs_pre.append(matrix_y[testcase_num])
    #print(np.array(inputs_pre).shape)
    inputs = torch.FloatTensor(np.array(inputs_pre))
    print(inputs.shape)
    INUNITS = len(inputs_pre[0])
    TESTNUM = minority_num

    if inputs_pre == 0:
        print("skip！", nums[version])
        sys.exit()
    dataset = torch.Tensor(inputs).float()
    num_steps = 100

    # 制定每一步的beta
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
           alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
           == one_minus_alphas_bar_sqrt.shape  # 确保所有列表长度一致
    print("all the same shape", betas.shape)

    # 计算任意时刻的x采样值，基于x_0和重参数化
    def q_x(x_0, t):
        """可以基于x[0]得到任意时刻t的x[t]"""
        # x_0：dataset，原始(10000, 2)的数据点集
        # t: torch.tensor([i]),i为采样几次
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声

    # num_shows = 20
    # fig, axs = plt.subplots(2, 10, figsize=(28, 3))
    # plt.rc('text', color='black')

    # # 共有10000个点，每个点包含两个坐标
    # # 生成100步以内每隔5步加噪声后的图像
    # for i in range(num_shows):
    #     j = i // 10
    #     k = i % 10
    #     q_i = q_x(dataset, torch.tensor([i * num_steps // num_shows]))  # 生成t时刻的采样数据
    #     axs[j, k].scatter(q_i[:, 0], q_i[:, 1], color='red', edgecolor='white')
    #     axs[j, k].set_axis_off()
    #     axs[j, k].set_title('$q(\mathbf{x}_{' + str(i * num_steps // num_shows) + '})$')
    class MLPDiffusion(nn.Module):
        def __init__(self, n_steps, num_units=512):
            super(MLPDiffusion, self).__init__()

            self.linears = nn.ModuleList(
                [
                    nn.Linear(INUNITS, num_units),
                    nn.ReLU(),
                    nn.Linear(num_units, num_units),
                    nn.ReLU(),
                    nn.Linear(num_units, num_units),
                    nn.ReLU(),
                    nn.Linear(num_units, INUNITS),
                ]
            )
            self.step_embeddings = nn.ModuleList(
                [
                    nn.Embedding(n_steps, num_units),
                    nn.Embedding(n_steps, num_units),
                    nn.Embedding(n_steps, num_units),
                ]
            )

        def forward(self, x, t):
            #  x = x_0
            for idx, embedding_layer in enumerate(self.step_embeddings):
                t_embedding = embedding_layer(t)
                x = self.linears[2 * idx](x)
                x += t_embedding
                x = self.linears[2 * idx + 1](x)

            x = self.linears[-1](x)

            return x

    # nn.Embedding：https://blog.csdn.net/qq_39540454/article/details/115215056
    # 使用示例：输出维度是2，输入是x和step
    # model = MLPDiffusion(num_steps)
    # output = model(x,step)


    seed = 1234

    class EMA():  # 不知道在哪里用了
        """构建一个参数平滑器"""

        def __init__(self, mu=0.01):
            self.mu = mu
            self.shadow = {}

        def register(self, name, val):
            self.shadow[name] = val.clone()

        def __call__(self, name, x):
            assert name in self.shadow
            new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
            self.shadow[name] = new_average.clone()
            return new_average

    print('Training model...')
    batch_size = 10000
    print(dataset.shape)
    if len(dataset)%2==1:
        dataset = dataset[:-1,:]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 2000
    model = MLPDiffusion(num_steps)  # 输出维度是2，输入是x和step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    gshape=np.empty((gnum,len(matrix_x[0])))
    #print(gshape.shape,20)
    x_seq = p_sample_loop(model, gshape.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
    x_seq=x_seq[0].detach().numpy()
    # threshold = float(np.mean(x_seq))
    #print(threshold)
    x_seq = (x_seq >= 0.5).astype(int)
    #print(x_seq.shape,2)
    path_w = version_dir / "matrix_diffusion.csv"
    print(path_w)
    save_matrix_as_csv(x_seq, version_dir, path_w, minority_label)
    print("generate complete")



if __name__ == "__main__":
    main('clac','2-f')
