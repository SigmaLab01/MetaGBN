import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from utils import *
import numpy as np
import os
# from ../PGBN_tool import PGBN_sampler
import torch.nn.functional as F

class GBN_model(nn.Module):
    def __init__(self, args, train_flage=True):
        super(GBN_model, self).__init__()

        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(1.0).float()
        self.wei_shape = torch.tensor(1e-1).float()

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.topic_size = args.topic_size
        self.topic_size = [self.vocab_size] + self.topic_size
        self.layer_num = len(self.topic_size) - 1
        self.embed_size = args.embed_size

        # document-level encoder hidden feature generation
        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in range(self.layer_num)])

        h_encoder = [Conv1D(self.hidden_size, 1, self.vocab_size)]
        for i in range(self.layer_num - 1):
            h_encoder.append(Conv1D(self.hidden_size, 1, self.hidden_size))
        self.h_encoder = nn.ModuleList(h_encoder)

        h_encoder_2 = [Conv1D(self.hidden_size, 1, self.vocab_size)]
        for i in range(self.layer_num - 1):
            h_encoder_2.append(Conv1D(self.hidden_size, 1, self.hidden_size))
        self.h_encoder_2 = nn.ModuleList(h_encoder_2)

        shape_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.topic_size[i + 1] + self.hidden_size)
                         for i in range(self.layer_num - 1)]
        shape_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size + self.topic_size[self.layer_num]))
        self.shape_encoder = nn.ModuleList(shape_encoder)

        scale_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.topic_size[i + 1] + self.hidden_size)
                         for i in range(self.layer_num - 1)]
        scale_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size + self.topic_size[self.layer_num]))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        if train_flage:
            self.batch_size = args.train_batch_size
            self.sup_sample_size = args.sup_sample_size
            self.qry_sample_size = args.qry_sample_size
        else:
            self.batch_size = args.test_batch_size
            self.sup_sample_size = args.sup_sample_size
            self.qry_sample_size = 3

        # set-level encoder latent variable generation
        Statistic_Network = [InvarientNet(sample_size=self.sup_sample_size,
                                          hidden_dim=self.hidden_size,
                                          hidden_c=self.topic_size[i + 1]) for i in range(self.layer_num - 1)]
        Statistic_Network.append(InvarientNet(sample_size=self.sup_sample_size,
                                              hidden_dim=self.hidden_size, hidden_c=self.topic_size[self.layer_num]))
        self.StatisticNetwork = nn.ModuleList(Statistic_Network)

        # global-level latent variable generation
        Get_embedding = [Init_Embedding(self.topic_size[i], self.embed_size) for i in range(len(self.topic_size))]
        self.Get_embedding = nn.ModuleList(Get_embedding)

        # set-level encoder latent variable generation
        StatisticNetworkAlpha = [StatisticAlpha(sample_size=self.sup_sample_size, hidden_dim=self.hidden_size,
                                                        topic_size=self.topic_size[i + 1], att_dim=self.embed_size) for i in range(self.layer_num)]
        self.StatisticNetworkAlpha = nn.ModuleList(StatisticNetworkAlpha)

        decoder = [GaussSoftmaxV3(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.cuda()))

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num = 1):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def reparameterize_2(self, Wei_shape_res, Wei_scale, Sample_num = 20):
        # sample one
        eps = torch.cuda.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        theta = torch.max(theta, self.real_min.cuda())
        return torch.mean(theta, dim=0, keepdim=False)

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))
        return - likelihood / x.shape[1]

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.cuda() * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.cuda() + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def kl_diagnormal_stdnormal(self, mean, logvar):
        a = mean ** 2
        b = torch.exp(logvar)
        c = -1
        d = -logvar
        return 0.5 * torch.sum(a + b + c + d)

    def kl_diagnormal_diagnormal(self, q_mean, q_logvar, p_mean, p_logvar):
        # Ensure correct shapes since no numpy broadcasting yet
        p_mean = p_mean.expand_as(q_mean)
        p_logvar = p_logvar.expand_as(q_logvar)
        a = p_logvar
        b = - 1
        c = - q_logvar
        d = ((q_mean - p_mean) ** 2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
        return 0.5 * torch.sum(a + b + c + d)

    def test_ppl(self, x, y, set_k_rec=None, set_l=None, set_l_tmp=None,
                  set_embed_mu=None, set_embed_logsigma=None, global_embed_mu=None, global_embed_logsigma=None):
        x_rec = self.inference(x, set_k_rec, set_l, set_l_tmp, set_embed_mu, set_embed_logsigma, global_embed_mu, global_embed_logsigma)
        x_2 = x_rec / (x_rec.sum(0) + real_min)
        ppl = y * torch.log(x_2.T + real_min) / -y.sum()
        return ppl.sum().exp()

    def forward(self, x, set_k_rec=None, set_l=None, set_l_tmp=None, set_phi_c=None,
                set_embed_mu=None, set_embed_logsigma=None, global_embed_mu=None, global_embed_logsigma=None):
        hidden_list = [0] * self.layer_num
        set_c = [0] * self.layer_num
        theta = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num
        kl_loss = [0] * self.layer_num
        loss = 0
        likelihood = 0

        x = x.view(-1, self.vocab_size)

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t-1])))
            hidden_list[t] = hidden

        for t in range(self.layer_num):
            # context_c[t] = context_l_tmp[t].repeat(1, self.qry_sample_size).view(self.qry_sample_size * self.batch_size, -1)
            set_c[t] = self.reparameterize(set_k_rec[t].repeat(1, self.qry_sample_size).view(self.qry_sample_size * self.batch_size, -1),
                                               set_l[t].repeat(1, self.qry_sample_size).view(self.qry_sample_size * self.batch_size, -1))

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                hidden_prior = torch.cat((hidden_list[t], set_c[t]), 1)
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_prior)),
                                       self.real_min.cuda())      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_prior)), self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))
                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                if t == 0:
                    phi_theta[t] = self.decoder[t](theta[t], t, global_embed_mu[t], global_embed_logsigma[t],
                                                               set_embed_mu[t], set_embed_logsigma[t])
                else:
                    phi_theta[t] = self.decoder[t](theta[t], t, set_embed_mu[t-1], set_embed_logsigma[t-1],
                                                               set_embed_mu[t], set_embed_logsigma[t])
            else:
                hidden_phitheta = torch.cat((hidden_list[t], phi_theta[t+1].permute(1, 0), set_c[t]), 1)
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.cuda())  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))
                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                if t == 0:
                    phi_theta[t] = self.decoder[t](theta[t], t, global_embed_mu[t], global_embed_logsigma[t], set_embed_mu[t], set_embed_logsigma[t])
                else:
                    phi_theta[t] = self.decoder[t](theta[t], t, set_embed_mu[t-1], set_embed_logsigma[t-1], set_embed_mu[t], set_embed_logsigma[t])

        for t in range(self.layer_num + 1):
            if t == 0:
                loss += self.compute_loss(x.permute(1, 0), phi_theta[t])
                likelihood = loss.item()

            elif t == self.layer_num:
                loss += 0.8 * self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).cuda(), torch.tensor(1.0, dtype=torch.float32).cuda(),
                                       set_k_rec[t - 1].permute(1, 0), set_l[t - 1].permute(1, 0))
                loss += 0.8 * self.KL_GamWei(set_c[t - 1].permute(1, 0), torch.tensor(1.0, dtype=torch.float32).cuda(),
                                       k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
            else:
                if t != 1:
                   loss += 0.8 * self.KL_GamWei(set_phi_c[t], torch.tensor(1.0, dtype=torch.float32).cuda(),
                                          set_k_rec[t - 1].repeat(1, self.sup_sample_size).view(self.batch_size * self.sup_sample_size, -1).permute(1, 0),
                                          set_l[t - 1].repeat(1, self.sup_sample_size).view(self.batch_size * self.sup_sample_size, -1).permute(1, 0))
                loss += 0.8 * self.KL_GamWei(phi_theta[t] + set_l_tmp[t - 1].repeat(1, self.qry_sample_size).view(self.qry_sample_size * self.batch_size, -1).permute(1, 0),
                                       torch.tensor(1.0, dtype=torch.float32).cuda(),  k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))

        # for t in range(self.layer_num + 1):
        #     loss += 0.0001 * self.kl_diagnormal_stdnormal(global_embed_mu[t], torch.tensor(0.0, dtype=torch.float32).cuda())
        #     if t > 0:
        #         loss += 0.0001 * self.kl_diagnormal_diagnormal(set_embed_mu[t - 1], torch.tensor(0.0, dtype=torch.float32).cuda(),
        #                                               global_embed_mu[t].unsqueeze(0),
        #                                               torch.tensor(0.0, dtype=torch.float32).cuda())
        return loss, likelihood

    def inference_prior(self, x):
        hidden_list = [0] * self.layer_num
        set_k_rec = [0] * self.layer_num
        set_l = [0] * self.layer_num
        set_l_tmp = [0] * self.layer_num
        set_c = [0] * self.layer_num
        set_phi_c = [0] * self.layer_num
        set_embed_mu = [0] * (self.layer_num)
        set_embed_logsigma = [0] * (self.layer_num)
        global_embed_mu = [0] * (self.layer_num + 1)
        global_embed_logsigma = [0] * (self.layer_num + 1)

        x = x.view(-1, self.vocab_size)

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t - 1])))
            hidden_list[t] = hidden

        for t in range(len(self.topic_size)):
            alpha_mu, alpha_logsigma = self.Get_embedding[t]()
            global_embed_mu[t] = alpha_mu
            global_embed_logsigma[t] = alpha_logsigma

        for t in range(self.layer_num - 1, -1, -1):
             set_k_rec[t], set_l[t], set_l_tmp[t] = self.StatisticNetwork[t](hidden_list[t])
             set_c[t] = set_l_tmp[t].repeat(1, self.sup_sample_size).view(self.batch_size * self.sup_sample_size, -1)  # self.reparameterize(set_k_rec[t], set_l[t])

             if t != 0:
                  set_phi_c[t] = self.decoder[t](set_c[t].permute(1, 0), t, global_embed_mu[t], global_embed_logsigma[t],
                                                   global_embed_mu[t + 1], global_embed_logsigma[t + 1],  global_flag=True)
        # set-level global latent variable inference
        for t in range(self.layer_num):
            alpha_mu, alpha_logsigma = self.StatisticNetworkAlpha[t](hidden_list[t], global_embed_mu[t + 1])
            set_embed_mu[t] = alpha_mu
            set_embed_logsigma[t] = alpha_logsigma

        return set_k_rec, set_l, set_l_tmp, set_phi_c, set_embed_mu, set_embed_logsigma, global_embed_mu, global_embed_logsigma

    def inference(self, x, set_k_rec=None, set_l=None, set_l_tmp=None,
                  set_embed_mu=None, set_embed_logsigma=None, global_embed_mu=None, global_embed_logsigma=None):
        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num
        set_c = [0] * self.layer_num

        x = x.view(-1, self.vocab_size)

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t - 1])))
            hidden_list[t] = hidden

        for t in range(self.layer_num):
            set_c[t] = set_l_tmp[t].repeat(1, self.sup_sample_size).view(self.sup_sample_size, -1)

        # document-level latent variable inference
        for t in range(self.layer_num - 1, -1, -1):
            if t == self.layer_num - 1:
                hidden_prior = torch.cat((hidden_list[t], set_c[t]), 1)
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_prior)), self.real_min.cuda())  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_prior)), self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))
                theta[t] = l_tmp[t].permute(1, 0)
                if t == 0:
                    phi_theta[t] = self.decoder[t](theta[t], t, global_embed_mu[t], global_embed_logsigma[t],
                                                   set_embed_mu[t], set_embed_logsigma[t])
                else:
                    phi_theta[t] = self.decoder[t](theta[t], t, set_embed_mu[t - 1], set_embed_logsigma[t - 1],
                                                   set_embed_mu[t], set_embed_logsigma[t])
            else:
                hidden_phitheta = torch.cat((hidden_list[t], phi_theta[t + 1].permute(1, 0), set_c[t]), 1)
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.cuda())  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.cuda())
                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)),
                                     self.real_min.cuda())
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))
                theta[t] = l_tmp[t].permute(1, 0)
                if t == 0:
                    phi_theta[t] = self.decoder[t](theta[t], t, global_embed_mu[t], global_embed_logsigma[t],
                                                   set_embed_mu[t], set_embed_logsigma[t])
                else:
                    phi_theta[t] = self.decoder[t](theta[t], t, set_embed_mu[t - 1], set_embed_logsigma[t - 1],
                                                   set_embed_mu[t], set_embed_logsigma[t])
        return phi_theta[0]
