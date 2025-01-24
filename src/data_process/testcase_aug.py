import csv
import sys
import pandas as pd
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data_process.pre_data import csv2txt
from paths import Paths


def save_matrix_as_csv(data, version_dir, path_w, label):
    df = pd.read_csv(version_dir / "matrix.csv")
    header_list = list(df.columns)[0:-1]
    header_list.append('error')

    with open(path_w, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        data = np.insert(data, -1, values=label, axis=1)
        for d in data:
            writer.writerow(d)


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
    else:
        minority_label = 0
        minority_num = pass_num
    for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == minority_label:
            inputs_pre.append(matrix_x[testcase_num])

    inputs = torch.FloatTensor(np.array(inputs_pre))
    INUNITS = len(inputs_pre[0])
    TESTNUM = minority_num
    if inputs_pre == 0:
        print("skip！", nums[version])
        sys.exit()
    """ 
       Definition discriminator
    """

    class discriminator(nn.Module):
        def __init__(self):
            super(discriminator, self).__init__()
            self.dis = nn.Sequential(
                nn.Linear(INUNITS, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.dis(x)
            return x

    """ 
    Definition generator
    """

    class generator(nn.Module):
        def __init__(self):
            super(generator, self).__init__()
            self.gen = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, INUNITS),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.gen(x)
            return x

    """ 
    create object
    """
    G = generator()
    D = discriminator()

    device = torch.device("cuda")
    D = D.to(device)
    G = G.to(device)
    """ 
    train discriminator
    """
    print("start training")
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
    for epoch in range(num_epoch):
        real_testcase = inputs.to(device)
        real_label = Variable(torch.ones(TESTNUM)).to(device)
        fake_label = Variable(torch.zeros(TESTNUM)).to(device)
        real_label_new = []
        for item in real_label:
            temp = []
            temp.append(item)
            real_label_new.append(temp)
        real_label_new = torch.tensor(real_label_new).to(device)
        real_label = real_label_new
        fake_label_new = []
        for item in fake_label:
            temp = []
            temp.append(item)
            fake_label_new.append(temp)
        fake_label_new = torch.tensor(fake_label_new).to(device)

        real_out = D(real_testcase)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out
        z = Variable(torch.randn(TESTNUM, z_dimension)).to(device)
        fake_testcase = G(z)
        fake_out = D(fake_testcase)
        d_loss_fake = criterion(fake_out,
                                fake_label_new)
        fake_scores = fake_out
        """ 
        loss function and optimization
        """
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()  # Update parameters

        """ 
        train generator
        """

        z = Variable(torch.randn(TESTNUM, z_dimension)).to(device)
        fake_testcase = G(z)
        output = D(fake_testcase)
        g_loss = criterion(output, real_label)
        """
        bp and optimize
        """
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if epoch % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                real_scores.data.mean(), fake_scores.data.mean()
            ))

    z = Variable(torch.randn((TESTNUM_TOTAL - TESTNUM * 2), z_dimension)).to(device)
    fake_testcase = G(z)
    fake_testcase = fake_testcase.cpu().detach()
    fake_testcase_numpy = fake_testcase.numpy()
    print(fake_testcase_numpy.shape)

    for item in fake_testcase_numpy:
        for element_num in range(len(item)):
            if item[element_num] <= 0.5:
                item[element_num] = 0
            else:
                item[element_num] = 1
    path_w = version_dir / "matrix_gan.csv"
    save_matrix_as_csv(fake_testcase_numpy, version_dir, path_w, minority_label)
    print("generate complete")


if __name__ == "__main__":
    main()
