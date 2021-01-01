import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
import matplotlib.pyplot as plt
import torchsnooper
import math
from prepareData import *


class PositionalEncoding(nn.Module):
    """
    正弦编码
    Args:
       dim (int): embedding size
    """

    def __init__(self, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        return emb


"""""""""
网络组件
"""""""""


class ResRGLU(nn.Module):
    def __init__(self, input_channel, output_channel, k_size, dilation):
        padding = int(((k_size - 1) * dilation) / 2)
        super(ResRGLU, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, output_channel, k_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(input_channel, output_channel, k_size, padding=padding, dilation=dilation)
        self.b1 = nn.BatchNorm1d(output_channel)
        self.b2 = nn.BatchNorm1d(output_channel)

    def forward(self, input):
        x1_ = self.conv1(input)
        x2_ = self.conv2(input)
        x1 = self.b1(x1_)
        x2 = self.b2(x2_)
        x2__ = torch.sigmoid(x2)
        return x1 * x2__ + input


class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """

    def __init__(self, in_size, out_size, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x_ = self.conv1(x)
        if self.pad != 0:
            x_ = x_[..., :-self.pad]
        return x_


class ResCGLU(nn.Module):
    def __init__(self, input_channel, output_channel, k_size, dilation):
        super(ResCGLU, self).__init__()
        self.Cconv1 = CausalConv1d(input_channel, output_channel, k_size, dilation)
        self.Cconv2 = CausalConv1d(input_channel, output_channel, k_size, dilation)
        self.b1 = nn.BatchNorm1d(output_channel)
        self.b2 = nn.BatchNorm1d(output_channel)

    def forward(self, input):
        x1 = self.Cconv1(input)
        x2 = self.Cconv2(input)
        x1_ = self.b1(x1)
        x2_ = self.b2(x2)
        x2__ = torch.sigmoid(x2_)
        return x1_ * x2__ + input


class SrcEncoder(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(SrcEncoder, self).__init__()
        self.d = nn.Dropout(p=0.3).to(self.device)
        self.conv1 = nn.Conv1d(93, 256, 1, dilation=1).to(self.device)
        self.b = nn.BatchNorm1d(256).to(self.device)
        for i in range(3):
            for j in range(4):
                setattr(self, 'ResRGLU' + str(i) + str(j), ResRGLU(256, 256, 5, 3 ** j).to(self.device))
        self.conv2 = nn.Conv1d(256, 512, 1, dilation=1).to(self.device)

    def forward(self, input):
        output_ = self.b(self.conv1(self.d(input)))
        output1 = None
        for i in range(3):
            for j in range(4):
                n = i - 1 if j == 0 else i
                m = 3 if j == 0 else j - 1
                index = str(n) + str(m)
                if i == 2 and j == 3:
                    output1 = getattr(self, 'ResRGLU' + str(i) + str(j)).forward(getattr(self, 'output' + index))
                else:
                    setattr(self, 'output' + str(i) + str(j), getattr(self, 'ResRGLU' + str(i) + str(j)).forward(
                        output_ if i == 0 and j == 0 else getattr(self, 'output' + index)))
        output = self.conv2(output1)
        return output


class TrgEnc(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(TrgEnc, self).__init__()
        self.d = nn.Dropout(p=0.3).to(self.device)
        self.conv1 = CausalConv1d(93, 256, 1, 1).to(self.device)
        self.b = nn.BatchNorm1d(256).to(self.device)
        for i in range(3):
            for j in range(4):
                setattr(self, 'ResCGLU' + str(i) + str(j), ResCGLU(256, 256, 3, 3 ** j).to(self.device))
        self.conv2 = CausalConv1d(256, 256, 1, 1).to(self.device)

    def forward(self, input):
        output_ = self.b(self.conv1(self.d(input)))
        output1 = None
        for i in range(3):
            for j in range(4):
                n = i - 1 if j == 0 else i
                m = 3 if j == 0 else j - 1
                index = str(n) + str(m)
                if i == 2 and j == 3:
                    output1 = getattr(self, 'ResCGLU' + str(i) + str(j)).forward(getattr(self, 'output' + index))
                else:
                    setattr(self, 'output' + str(i) + str(j), getattr(self, 'ResCGLU' + str(i) + str(j)).forward(
                        output_ if i == 0 and j == 0 else getattr(self, 'output' + index)))
        output = self.conv2(output1)
        return output


class TrgRec(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(TrgRec, self).__init__()
        self.d = nn.Dropout(p=0.3).to(self.device)
        self.conv1 = nn.Conv1d(256, 256, 1, dilation=1).to(self.device)
        self.b = nn.BatchNorm1d(256).to(self.device)
        for i in range(3):
            for j in range(4):
                setattr(self, 'ResRGLU' + str(i) + str(j), ResRGLU(256, 256, 5, 3 ** j).to(self.device))
        self.conv2 = nn.Conv1d(256, 93, 1, dilation=1).to(self.device)

    def forward(self, input):
        output_ = self.b(self.conv1(self.d(input)))
        output1 = None
        for i in range(3):
            for j in range(4):
                n = i - 1 if j == 0 else i
                m = 3 if j == 0 else j - 1
                index = str(n) + str(m)
                if i == 2 and j == 3:
                    output1 = getattr(self, 'ResRGLU' + str(i) + str(j)).forward(getattr(self, 'output' + index))
                else:
                    setattr(self, 'output' + str(i) + str(j), getattr(self, 'ResRGLU' + str(i) + str(j)).forward(
                        output_ if i == 0 and j == 0 else getattr(self, 'output' + index)))
        output = self.conv2(output1)
        return output


class TrgDec(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(TrgDec, self).__init__()
        self.d = nn.Dropout(p=0.3).to(self.device)
        self.conv1 = CausalConv1d(256, 256, 1, 1).to(self.device)
        self.b = nn.BatchNorm1d(256).to(self.device)
        for i in range(3):
            for j in range(4):
                setattr(self, 'ResCGLU' + str(i) + str(j), ResCGLU(256, 256, 3, 3 ** j).to(self.device))
        self.conv2 = CausalConv1d(256, 93, 1, 1).to(self.device)

    def forward(self, input):
        output_ = self.b(self.conv1(self.d(input)))
        output1 = None
        for i in range(3):
            for j in range(4):
                n = i - 1 if j == 0 else i
                m = 3 if j == 0 else j - 1
                index = str(n) + str(m)
                if i == 2 and j == 3:
                    output1 = getattr(self, 'ResCGLU' + str(i) + str(j)).forward(getattr(self, 'output' + index))
                else:
                    setattr(self, 'output' + str(i) + str(j), getattr(self, 'ResCGLU' + str(i) + str(j)).forward(
                        output_ if i == 0 and j == 0 else getattr(self, 'output' + index)))
        output = self.conv2(output1)
        return output


# %%
"""
模型定义
"""


class Model_(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Model_, self).__init__()
        self.SrcEnc = SrcEncoder().to(self.device)
        self.TrgEnc = TrgEnc().to(self.device)
        self.TrgDec = TrgDec().to(self.device)
        self.TrgRec = TrgRec().to(self.device)
        self.debugA = []
        self.positionEncoding = PositionalEncoding(93)

    def forward(self, X, y):
        """
        前向传播
        :param X: 训练集
        :param y: 训练集标签
        :return:
        """
        self.train()
        X_ = X.view((1, X.size()[0], X.size()[1]))
        X = self.positionEncoding(X_)
        K_V = self.SrcEnc(X)
        Y = torch.cat([torch.zeros((y.size()[0], 1)).cuda(), y.clone().cuda()], dim=1)
        Y = Y.view((1, Y.size()[0], Y.size()[1]))
        Q = self.TrgEnc(Y)
        K_V = K_V.view(K_V.size()[1], K_V.size()[2])
        K, V = K_V.split(256, dim=0)
        Q = Q.view((Q.size()[1], Q.size()[2]))
        temp = (torch.t(K) @ Q) / np.sqrt(256)
        exp_temp = torch.exp(temp)
        exp_temp_sum = exp_temp.sum(dim=1).view(exp_temp.size()[0], 1)
        A = exp_temp / exp_temp_sum
        R = V @ A
        R = R.view(1, R.size()[0], R.size()[1])
        Y_dec = self.TrgDec(R)
        Y_rec = self.TrgRec(R)
        return Y_dec, Y_rec, A

    def prdict(self, input, N0: int, N1: int, debug):  # NO,N1 at the
        # nearest integers that correspond to 160[ms] and 320[ms]
        """
        用于使用模型预测
        :param input: 输入值
        :param N0:
        :param N1:
        :param debug:
        :return:
        """
        n_hat = 0
        Y = torch.zeros((1, 93, 2)).cuda()
        A = None
        X = self.positionEncoding(input)
        K_V = self.SrcEnc(X)
        if debug:
            self.debugA.append(K_V)
        K_V = K_V.view(K_V.size()[1], K_V.size()[2])
        K, V = K_V.split(256, dim=0)
        N = K.size()[0] - 1
        R = None
        for m in range(20):
            Q = self.TrgEnc(Y)
            Q = Q.view((Q.size()[1], Q.size()[2]))
            temp = (torch.t(K) @ Q) / np.sqrt(256)
            exp_temp = torch.exp(temp)
            exp_temp_sum = exp_temp.sum(dim=1).view(exp_temp.size()[0], 1)
            A = exp_temp / exp_temp_sum
            if m > 0:
                a = A[:, m]
                if max(0, n_hat - N0) != 0:
                    a[0:max(0, n_hat - N0)] = 0
                else:
                    a[0] = 0
                if min(n_hat + N1, N) != N:
                    a[min(n_hat + N1, N):N] = 0
                else:
                    a[N] = 0
                a = a / a.sum()
                a = a.view((A.size()[0], 1))
                A = torch.cat([A[:, :m].clone(), a], dim=1)
            n_hat = torch.argmax(A[:, m])
            R = V @ A
            R = R.view(1, R.size()[0], R.size()[1])
            Y = self.TrgDec(R)
            Y = Y.view(Y.size()[1], Y.size()[2])
            Y = torch.cat([torch.zeros((Y.size()[0], 1)).cuda(), Y.clone().cuda()], dim=1)
            Y = Y.view((1, Y.size()[0], Y.size()[1]))
        Y_hat = self.TrgRec(R)
        return Y, Y_hat, A

    def cal_Loss(self, Y, Y_hat, label, A):
        """
        计算Loss
        :param Y: [channel,time]
        :param Y_hat: [channel,time]
        :param label: [channel,time]
        :return: total_Loss
        """
        dec_Loss = self.cal_dec_Loss(Y, label)
        rec_Loss = self.cal_rec_Loss(Y_hat, label)
        dal_Loss, oal_Loss = self.cal_dal_oal_Loss(A, A.size()[0], A.size()[1])
        return dec_Loss + rec_Loss + 2000 * dal_Loss + 2000 * oal_Loss

    def cal_dec_Loss(self, Y, label):  # label.size() = (channel,Nt)
        Nt = label.size()[1]
        dec_Loss = torch.abs(Y[:, :Nt - 1] - label[:, 1:]).sum()
        rel_dec_Loss = dec_Loss / Nt
        return rel_dec_Loss

    def cal_rec_Loss(self, Y_hat, label):
        Nt = label.size()[1]
        rec_Loss = torch.abs(Y_hat[:, :Nt] - label)
        rec_Loss = rec_Loss / Nt
        return rec_Loss.sum()

    def cal_dal_oal_Loss(self, A, Ns: int, Nt: int):
        dal_W = self.generate_W(Ns, Nt, 0.3)
        dal_Loss = torch.abs((dal_W * A)).sum() / (Ns * Nt)
        oal_W = self.generate_W(Ns, Ns, 0.3)
        oal_Loss = torch.abs((oal_W * (A @ torch.t(A)))).sum() / (Ns ** 2)
        return dal_Loss, oal_Loss

    def generate_W(self, Ns: int, Nt: int, v):
        x = torch.full([Ns, Nt], Ns).cuda()
        y = torch.full([Ns, Nt], Nt).cuda()
        n = torch.range(0, Ns - 1).view(Ns, 1).cuda()
        m = torch.range(Nt - 1, 0, -1).view(1, Nt).cuda()
        result = x - torch.exp(-(n / x - m / y) ** 2 / (2 * (v ** 2)))
        return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_filedir = ""
    logf0s_mean, logf0s_std, mcs_mean, mcs_std = getDataForPrepare(X_filedir, 4)
    Y_filedir = ""
    Ylogf0s_mean, Ylogf0s_std, Ymcs_mean, Ymcs_std = getDataForPrepare(X_filedir, 4)
    X_wav_file = ""
    X = PrepareDate(X_wav_file, logf0s_mean, logf0s_std, mcs_mean, mcs_std)
    Y_wav_file = ""
    Y = PrepareDate(Y_wav_file, logf0s_mean, Ylogf0s_std, Ymcs_mean, Ymcs_std)
    model = Model_().to(device)
    max_epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)
    for epoch in range(max_epoch):
        print("----epoch:", epoch)
        Y_dec, Y_rec, A = model.forward(X, Y)
        loss = model.cal_Loss(Y_dec, Y_rec, Y, A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
