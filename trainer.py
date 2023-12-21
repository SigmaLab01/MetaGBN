import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from model import *
import matplotlib.pyplot as plt
from utils import gen_ppl_doc
import pickle
import numpy

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.model = GBN_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.sup_size = args.sup_sample_size
        self.qry_size = args.qry_sample_size

    def train(self, train_data_loader, test_data_loader):

        for epoch in range(self.epochs):
            self.model.cuda()
            self.model.train()

            self.model.batch_size = 19
            self.model.sup_sample_size = self.sup_size
            self.model.qry_sample_size = self.qry_size

            for s in range(20):
                accum_step = 5
                num_data = len(train_data_loader)

                for temp in range(accum_step):
                    for i, (train_data) in enumerate(train_data_loader):
                        train_data_sup = torch.tensor(train_data[:, :self.sup_size, :], dtype=torch.float).cuda()
                        train_data_qry = torch.tensor(train_data[:, :, :], dtype=torch.float).cuda()

                        set_k_rec, set_l, set_l_tmp, set_phi_c, \
                        set_embed_mu, set_embed_logsigma, global_embed_mu, global_embed_logsig = self.model.inference_prior(
                            train_data_sup)

                        loss, likelihood = self.model(train_data_qry, set_k_rec, set_l, set_l_tmp, set_phi_c,
                                                      set_embed_mu, set_embed_logsigma, global_embed_mu,
                                                      global_embed_logsig)

                        loss = loss / (accum_step * num_data)
                        likelihood = likelihood / (accum_step * num_data)
                        loss.backward()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if epoch % 10 == 0:
                print('epoch {}|{}, loss: {}, likelihood: {}'.format(epoch, self.epochs, loss.item(), likelihood))
                save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch}, self.save_path, True)
                self.vis_txt()

            if epoch % 10 == 0:
                print('epoch {}|{}, test_ikelihood,{}'.format(epoch, self.epochs, self.test(test_data_loader)))

    def test(self, data_loader):
        ppl = 0

        with open('dataset/20ng_statis.pkl', 'rb') as f:
            data = pickle.load(f)
        data_input = [data['test_class_data']]

        for i in range(len(data_input)):
            num = len(data_input[i]) // (self.sup_size + self.sup_size)

            self.model.batch_size = 1
            self.model.sup_sample_size = self.sup_size
            self.model.qry_sample_size = self.sup_size

            for iter in range(num):
                test_data = data_input[i][iter * (self.sup_size + self.sup_size):(iter + 1) * (self.sup_size + self.sup_size)]

                test_data_sup = numpy.array(test_data[:self.sup_size])
                test_data_sup = torch.from_numpy(test_data_sup).float().cuda()

                test_data_qry = test_data[self.sup_size:]
                test_qry_data_in, test_qry_data_out = gen_ppl_doc(test_data_qry)
                test_qry_data_in = torch.tensor(test_qry_data_in, dtype=torch.float).cuda()
                test_qry_data_out = torch.tensor(test_qry_data_out, dtype=torch.float).cuda()

                with torch.no_grad():
                    self.model.eval()
                    set_k_rec, set_l, set_l_tmp, set_phi_c, \
                    set_embed_mu, set_embed_logsigma, global_embed_mu, global_embed_logsigma = self.model.inference_prior(
                        test_data_sup)
                    ppl += self.model.test_ppl(test_qry_data_in, test_qry_data_out, set_k_rec, set_l, set_l_tmp,
                                             set_embed_mu, set_embed_logsigma, global_embed_mu,
                                             global_embed_logsigma).item() / (num * len(data_input))
        return ppl

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    def vis(self):
        # layer1
        w_1 = torch.mm(self.GBN_models[0].decoder.rho, torch.transpose(self.GBN_models[0].decoder.alphas, 0, 1))
        phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

        index1 = range(100)
        dic1 = phi_1[:, index1[0:49]]
        # dic1 = phi_1[:, :]
        fig7 = plt.figure(figsize=(10, 10))
        for i in range(dic1.shape[1]):
            tmp = dic1[:, i].reshape(28, 28)
            ax = fig7.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index1[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_2 = torch.mm(self.GBN_models[1].decoder.rho, torch.transpose(self.GBN_models[1].decoder.alphas, 0, 1))
        phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()
        index2 = range(49)
        dic2 = np.matmul(phi_1, phi_2[:, index2[0:49]])
        #dic2 = np.matmul(phi_1, phi_2[:, :])
        fig8 = plt.figure(figsize=(10, 10))
        for i in range(dic2.shape[1]):
            tmp = dic2[:, i].reshape(28, 28)
            ax = fig8.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index2[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_3 = torch.mm(self.GBN_models[2].decoder.rho, torch.transpose(self.GBN_models[2].decoder.alphas, 0, 1))
        phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()
        index3 = range(32)

        dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, index3[0:32]])
        #dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, :])
        fig9 = plt.figure(figsize=(10, 10))
        for i in range(dic3.shape[1]):
            tmp = dic3[:, i].reshape(28, 28)
            ax = fig9.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index3[i] + 1))
            ax.imshow(tmp)

        plt.show()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self):
        phi = []
        for t in range(self.layer_num):
            w_t = self.model.decoder[t].w[0, :, :].cpu().detach().numpy()
            phi.append(w_t)
        self.vision_phi(phi)