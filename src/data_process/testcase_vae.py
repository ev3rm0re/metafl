from torch import nn, optim
import torch
import torch.nn.functional as F
import sys
import numpy as np
from data_process.pre_data import csv2txt
from paths import Paths
import csv
import pandas as pd
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(program, version):
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
            #print(p)
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
    num_epoch = 100  # 1000
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
        gnum = pass_num - fail_num
    else:
        minority_label = 0
        minority_num = pass_num
        gnum = fail_num-pass_num
    for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == minority_label:
            inputs_pre.append(matrix_x[testcase_num])
    print(np.array(inputs_pre).shape)
    inputs = torch.FloatTensor(np.array(inputs_pre))

    INUNITS = len(inputs_pre[0])
    print(INUNITS)
    TESTNUM = minority_num
    if inputs_pre == 0:
        print("skip！", nums[version])
        sys.exit()
    """ 
       Definition discriminator
    """
    """模型和关于模型函数"""
    device = 'cuda'
    h_dim=800
    z_dim=100
    lr=0.0003
    epochs=15
    class VAE(nn.Module):
        def __init__(self, input_dim=INUNITS, h_dim=h_dim, z_dim=z_dim):
            # 调用父类方法初始化模块的state
            super(VAE, self).__init__()

            self.input_dim = input_dim
            self.h_dim = h_dim
            self.z_dim = z_dim

            # 编码器 ： [b, input_dim] => [b, z_dim]
            self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
            self.fc6=nn.Linear(h_dim,h_dim)
            self.fc7=nn.Linear(h_dim,h_dim)

            self.fc2 = nn.Linear(h_dim, z_dim)  # mu
            self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

            # 解码器 ： [b, z_dim] => [b, input_dim]

            self.fc4 = nn.Linear(z_dim, h_dim)
            self.fc5 = nn.Linear(h_dim, input_dim)
            self.fc8 = nn.Linear(h_dim, h_dim)
            self.fc9 = nn.Linear(h_dim, h_dim)
        def forward(self, x):

            batch_size = x.shape[0]

            x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

            # encoder
            mu, log_var = self.encode(x)
            # reparameterization trick
            sampled_z = self.reparameterization(mu, log_var)
            # decoder
            x_hat = self.decode(sampled_z)
            # reshape
            x_hat = x_hat.view(batch_size, self.input_dim)
            return x_hat, mu, log_var

        def encode(self, x):
            """
            encoding part
            :param x: input image
            :return: mu and log_var
            """
            h = F.relu(self.fc6(F.relu(self.fc7(F.relu(self.fc1(x))))))
            mu = self.fc2(h)
            log_var = self.fc3(h)

            return mu, log_var

        def decode(self, z):
            """
            Given a sampled z, decode it back to image
            """
            h = F.relu(self.fc8(F.relu(self.fc9(F.relu(self.fc4(z))))))
            x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
            return x_hat

        def reparameterization(self, mu, log_var):
            """
            Given a standard gaussian distribution epsilon ~ N(0,1),
            we can sample the random variable z as per z = mu + sigma * epsilon
            """
            sigma = torch.exp(log_var * 0.5)
            eps = torch.randn_like(sigma)
            return mu + sigma * eps  # *是点乘的意思

        def gener(self, gnum, z_dim,mu,log_var):
            with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
                '''测试随机生成的隐变量'''
                # 随机从隐变量的分布中取隐变量
                print('z_dim',z_dim)
                z = torch.randn(gnum, z_dim).to('cuda')  # 每一行是一个隐变量，总共有batch_size行
                # 对隐变量重构
                # lendata=len(mu)
                # mudata=torch.tensor([])
                # log_vardata=torch.tensor([])

                # while gnum>lendata:
                #     indices = torch.randperm(len(mu))[:gnum]
                #     print('indices',len(indices))
                #     mudata=mudata.append(mu[indices,:])
                #     print(mu.shape)
                #     indices1 = torch.randperm(len(log_var))[:gnum]
                #     log_vardata=log_vardata.append(log_var[indices1,:])
                #     gnum=gnum-lendata
                indices = torch.randint(0, len(mu), (gnum,))
                mu=mu[indices,:]
                indices1 = torch.randint(0, len(log_var), (gnum,))
                log_var = log_var[indices1, :]
                z=z*torch.exp(log_var * 0.5)+mu
                random_res = self.decode(z)
                # 保存重构结果
                random_res=np.array(random_res.detach().cpu())


                return random_res



    def loss_function(x_hat, x, mu, log_var):
        """
        Calculate the loss. Note that the loss includes two parts.
        :return: total loss, BCE and KLD of our model
        """
        # 1. the reconstruction loss.
        # We regard the MNIST as binary classification
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        BCE = criterion(x_hat, x)

        # 2. KL-divergence
        KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

        # 3. total loss
        loss = BCE + KLD
        return loss, BCE, KLD

    model = VAE().to(device)  # 创建VAE模型实例，并转移到GPU上去
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 初始化优化器，需要优化的是model的参数，learning rate设置为0.001
    x = inputs.to(device)
    print('x_shape',x.shape)
    print(INUNITS)
    for epoch in range(5000):

        # 前向传播
        x_hat, mu, log_var = model(x)  # 模型调用会自动调用model中的forward函数
        loss, BCE, KLD = loss_function(x_hat, x, mu, log_var)  # 计算损失值，即目标函数
        if epoch%1000==0:
            # print(mu)
            # print(log_var)
            print('epoch:',epoch,'loss:',loss.detach().cpu())
        # 后向传播
        optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
        loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
        optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了
    # print(z_dim)
    x_hat, mu, log_var = model(x)
    # print(mu)
    # print(log_var)
    array=model.gener(gnum,z_dim,mu,log_var)
    path_w = version_dir / "matrix_vae.csv"
    # print(path_w)
    save_matrix_as_csv(array, version_dir, path_w, minority_label)
    print("generate complete")



if __name__ == "__main__":
    main('clac','1-f')
