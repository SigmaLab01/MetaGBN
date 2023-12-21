import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import numpy as np

real_min = torch.tensor(1e-30)

def gen_ppl_doc(x, ratio=0.8):
    """
    inputs:
        x: N x V, np array,
        ratio: float or double,
    returns:
        x_1: N x V, np array, the first half docs whose length equals to ratio * doc length,
        x_2: N x V, np array, the second half docs whose length equals to (1 - ratio) * doc length,
    """
    import random
    x_1, x_2 = np.zeros_like(x), np.zeros_like(x)
    # indices_x, indices_y = np.nonzero(x)
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2020)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        l_2 = l[int(len(l) * ratio):]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
        for l2_value in l_2:
            x_2[doc_idx][l2_value] += 1
    return x_1, x_2

def log_max(x):
    return torch.log(torch.max(x, real_min.cuda()))

def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    eulergamma = torch.tensor(0.5772, dtype=torch.float32)

    part1 = eulergamma.cuda() * (1 - 1 / Wei_shape) + log_max(
        Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)

    part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma.cuda() / Wei_shape)

    part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))

    KL = part1 + part2 + part3
    return KL

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf).cuda()
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class DeepConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(DeepConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf).cuda())

            w2 = torch.empty(nf, nf).cuda()
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf).cuda())

        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            rx = x
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = x + rx
        else:
            raise NotImplementedError
        return x

class ResConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(ResConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf).cuda())

            w2 = torch.empty(nx, nf).cuda()
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            rx = x
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = rx + x
        else:
            raise NotImplementedError
        return x

class Conv1DSoftmax(nn.Module):
    def __init__(self, voc_size, topic_size):
        super(Conv1DSoftmax, self).__init__()

        w = torch.empty(voc_size, topic_size).cuda()
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)

    def forward(self, x):
        w = torch.softmax(self.w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

class Conv1DSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, last_layer=None):
        super(Conv1DSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        if last_layer is None:
            w1 = torch.empty(self.voc_size, self.emb_size).cuda()
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)
        else:
            w1 = torch.empty(self.voc_size, self.emb_size).cuda()
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)

        w2 = torch.empty(self.topic_size, self.emb_size).cuda()
        nn.init.normal_(w2, std=0.02)
        self.alphas = Parameter(w2)

    def forward(self, x, t):
        if t == 0:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        else:
            w = torch.mm(self.rho.detach(), torch.transpose(self.alphas, 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

import itertools
import math

class GaussSoftmaxV3(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size):
        super(GaussSoftmaxV3, self).__init__()
        self.vocab_size = voc_size
        self.topic_size = topic_size
        self.embed_dim = emb_size
        self.sigma_min = 0.1
        self.sigma_max = 10.0
        self.C = 2.0
        self.lamda = 1.0

    def el_energy_1(self, mu_v, mu_t, sigma_v, sigma_t):
        mu_v = mu_v.unsqueeze(0).unsqueeze(2)  # 1 * vocab * 1  * embed
        sigma_v = sigma_v.unsqueeze(0).unsqueeze(2)  # 1 * vocab * 1 * embed
        mu_t = mu_t.unsqueeze(1)  # batch_size * 1 * topic  * embed
        sigma_t = sigma_t.unsqueeze(1)   # batch_size * 1 * topic  * embed

        det_fac = torch.sum(torch.log(sigma_v + sigma_t), 3)    # V * K
        diff_mu = torch.sum((mu_v - mu_t) ** 2 / (sigma_v + sigma_t), 3)  # V * K
        return -0.5 * (det_fac + diff_mu + self.embed_dim * math.log(2 * math.pi))   # V * K

    def el_energy_2(self, mu_v, mu_t, sigma_v, sigma_t):
        mu_v = mu_v.unsqueeze(2)  # 1 * vocab * 1  * embed
        sigma_v = sigma_v.unsqueeze(2)  # 1 * vocab * 1 * embed
        mu_t = mu_t.unsqueeze(1)  # batch_size * 1 * topic  * embed
        sigma_t = sigma_t.unsqueeze(1)   # batch_size * 1 * topic  * embed

        det_fac = torch.sum(torch.log(sigma_v + sigma_t), 3)    # V * K
        diff_mu = torch.sum((mu_v - mu_t) ** 2 / (sigma_v + sigma_t), 3)  # V * K
        return -0.5 * (det_fac + diff_mu + self.embed_dim * math.log(2 * math.pi))   # V * K

    def el_energy_3(self, mu_v, mu_t, sigma_v, sigma_t):
        mu_v = mu_v.unsqueeze(1)  # vocab * 1  * embed
        sigma_v = sigma_v.unsqueeze(1)  #  vocab * 1 * embed
        mu_t = mu_t.unsqueeze(0)  #  1 * topic  * embed
        sigma_t = sigma_t.unsqueeze(0)  #  1 * topic  * embed

        det_fac = torch.sum(torch.log(sigma_v + sigma_t), 2)  # V * K
        diff_mu = torch.sum((mu_v - mu_t) ** 2 / (sigma_v + sigma_t), 2)  # V * K
        return -0.5 * (det_fac + diff_mu + self.embed_dim * math.log(2 * math.pi))  # V * K

    def kl_diagnormal_stdnormal(self, mean, logvar):
        a = mean ** 2
        b = torch.exp(logvar)
        c = -1
        d = -logvar
        return 0.5 * torch.mean(a + b + c + d)

    def kl_energy(self, mu_v, mu_t, sigma_v, sigma_t):
        mu_v = mu_v.unsqueeze(1)  # vocab * 1  * embed
        sigma_v = sigma_v.unsqueeze(1)  # vocab * 1 * embed
        mu_t = mu_t.unsqueeze(0).unsqueeze(1)  # 1 * topic  * embed
        sigma_t = sigma_t.unsqueeze(0).unsqueeze(1)  # 1 * topic  * embed

        det_fac = torch.sum(torch.log(sigma_t), 3) - torch.sum(torch.log(sigma_v), 3)  # vocab * topic
        trace_fac = torch.sum(sigma_v / sigma_t, 3)  # vocab * * topic
        diff_mu = torch.sum((mu_v - mu_t) ** 2 / sigma_t, 3)  # vocab * * topic
        return torch.mean(0.5 * (det_fac - self.embed_dim + trace_fac + diff_mu))  # vocab * topic

    def forward(self, x, t, mu_v, log_sigma_v, mu_c, log_sigma_c, global_flag=False):
        for p in itertools.chain(log_sigma_v, log_sigma_c):
            p.data.clamp_(math.log(self.sigma_min), math.log(self.sigma_max))

        for p in itertools.chain(mu_v, mu_c):
            p.data.clamp_(-math.sqrt(self.C), math.sqrt(self.C))

        if global_flag:
            # w = torch.softmax(self.el_energy_3(mu_v, mu_c, torch.exp((log_sigma_v)), torch.exp(log_sigma_c)), dim=0)
            w = torch.softmax(torch.mm(mu_v, torch.transpose(mu_c, 0, 1)), dim=0)
            x = torch.mm(w, x.view(-1, x.size(-1)))
        else:
            if t == 0:
                self.w = torch.softmax(torch.bmm(mu_v.unsqueeze(0).repeat(mu_c.shape[0], 1, 1), torch.transpose(mu_c, 1, 2)), dim=1)
                #self.w = torch.softmax(self.el_energy_1(mu_v, mu_c, torch.exp((log_sigma_v)), torch.exp(log_sigma_c)), dim=1)
            else:
                self.w = torch.softmax(torch.bmm(mu_v.detach(), torch.transpose(mu_c, 1, 2)), dim=1)
               # self.w = torch.softmax(self.el_energy_2(mu_v.detach(), mu_c, torch.exp((log_sigma_v.detach())), torch.exp(log_sigma_c)), dim=1)
            x = torch.bmm(self.w, x.transpose(0, 1).contiguous().view(self.w.shape[0], -1, self.w.shape[2]).transpose(1, 2)).transpose(1, 2).contiguous().view(x.shape[1], -1).transpose(0, 1)
        return x


class GaussSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size):
        super(GaussSoftmaxEtm, self).__init__()
        self.vocab_size = voc_size
        self.topic_size = topic_size
        self.embed_dim = emb_size
        self.sigma_min = 0.1
        self.sigma_max = 10.0
        self.C = 2.0

        self.w = 0

        # Model
        self.mu = nn.Embedding(self.vocab_size, self.embed_dim)
        self.log_sigma = nn.Embedding(self.vocab_size, self.embed_dim)

        self.mu_c = nn.Embedding(self.topic_size, self.embed_dim)
        self.log_sigma_c = nn.Embedding(self.topic_size, self.embed_dim)

    def el_energy(self, mu_i, mu_j, sigma_i, sigma_j):
        """
        :param mu_i: mu of word i: [batch, embed]
        :param mu_j: mu of word j: [batch, embed]
        :param sigma_i: sigma of word i: [batch, embed]
        :param sigma_j: sigma of word j: [batch, embed]
        :return: the energy function between the two batchs of  data: [batch]
        """

        # assert mu_i.size()[0] == mu_j.size()[0]

        det_fac = torch.sum(torch.log(sigma_i + sigma_j), 1)
        diff_mu = torch.sum((mu_i - mu_j) ** 2 / (sigma_j + sigma_i), 1)
        return -0.5 * (det_fac + diff_mu + self.embed_dim * math.log(2 * math.pi))

    def forward(self, x, t):
        for p in itertools.chain(self.log_sigma.parameters(),
                                 self.log_sigma_c.parameters()):
            p.data.clamp_(math.log(self.sigma_min), math.log(self.sigma_max))

        for p in itertools.chain(self.mu.parameters(),
                                 self.mu_c.parameters()):
            p.data.clamp_(-math.sqrt(self.C), math.sqrt(self.C))

        w = torch.zeros((self.vocab_size, self.topic_size)).cuda()
        for i in range(self.topic_size):
            log_el_energy = self.el_energy(self.mu.weight, self.mu_c.weight[i, :], torch.exp(self.log_sigma.weight), torch.exp(self.log_sigma_c.weight[i, :]))
            w[:, i] = torch.softmax(log_el_energy, dim=0)

        self.w = w
        x = torch.mm(w, x.view(w.shape[0], -1, x.size(-1)))
        return x

def variable_para(shape, device='cuda'):
    w = torch.empty(shape, device=device)
    nn.init.normal_(w, std=0.02)
    return torch.tensor(w, requires_grad=True)

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


class InvarientNet(nn.Module):
    def __init__(self, sample_size, hidden_dim,  hidden_c):
        super(InvarientNet, self).__init__()
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(1.0).float()
        self.wei_shape = torch.tensor(1e-1).float()

        self.sample_size = sample_size
        self.hidden_dim = hidden_dim

        self.shape_encoder = Conv1D(hidden_c, 1,  hidden_dim)
        self.scale_encoder = Conv1D(hidden_c, 1,  hidden_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, h):
        e = self.dropout(self.pool(h))
        k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder(e)),
                               self.real_min.cuda())  # k_rec = 1/k
        context_k_rec = torch.min(k_rec_temp, self.wei_shape_max.cuda())
        l_tmp = torch.max(torch.nn.functional.softplus(self.scale_encoder(e)), self.real_min.cuda())
        context_l = l_tmp / torch.exp(torch.lgamma(1 + context_k_rec))
        return context_k_rec, context_l, l_tmp

    def pool(self, e):
        e = e.view(-1, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(-1, self.hidden_dim)
        return e.detach()


class StatisticAlpha(nn.Module):
    def __init__(self, sample_size, hidden_dim, topic_size, att_dim):
        super(StatisticAlpha, self).__init__()
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(1.0).float()
        self.wei_shape = torch.tensor(1e-1).float()

        self.sample_size = sample_size
        self.hidden_dim = hidden_dim
        self.topic_size = topic_size
        self.att_dim = att_dim

        self.alpha_mu = Conv1D(topic_size * att_dim, 1, hidden_dim)
        self.alpha_logsigma = Conv1D(topic_size * att_dim, 1, hidden_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, h, global_embedding):
        e = self.dropout(self.pool(h))
        alpha_mu = self.alpha_mu(e).view(-1, self.topic_size, self.att_dim) + self.dropout(global_embedding).squeeze(0)
        alpha_logsigma = self.alpha_logsigma(e).view(-1, self.topic_size, self.att_dim)
        return alpha_mu, alpha_logsigma

    def pool(self, e):
        e = e.view(-1, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(-1, self.hidden_dim)
        return e.detach()


class Init_Embedding(nn.Module):
    def __init__(self, voc_size, embed_size):
        super(Init_Embedding, self).__init__()
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(1.0).float()
        self.wei_shape = torch.tensor(1e-1).float()

        self.voc_size = voc_size
        self.embed_size = embed_size

        w3 = torch.empty(self.voc_size, self.embed_size).cuda()
        nn.init.normal_(w3, std=0.02)
        self.voc_mu = Parameter(w3)

        w4 = torch.empty(self.voc_size, self.embed_size).cuda()
        nn.init.normal_(w4, std=0.02)
        self.voc_logsigma = Parameter(w4)

    def forward(self):
        return self.voc_logsigma, self.voc_logsigma