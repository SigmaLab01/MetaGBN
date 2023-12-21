import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from mydataset import *
from model import *
import pickle
from trainer import GBN_trainer
import argparse
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument('--train-batch-size', type=int, default=19, help="models used.")
parser.add_argument('--test-batch-size', type=int, default=5, help="models used.")
parser.add_argument('--train-sample-size', type=int, default=15, help="models used.")
parser.add_argument('--test-sample-size', type=int, default=3, help="models used.")
parser.add_argument('--sup-sample-size', type=int, default=5, help="support size")
parser.add_argument('--qry-sample-size', type=int, default=15, help="query size")
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--n_updates', type=int, default=1, help='parameter of gbn')
parser.add_argument('--MBratio', type=int, default=100, help='parameter of gbn')
parser.add_argument('--topic_size', type=list, default=[256, 128, 64], help='Number of units in hidden layer 1.') # [512, 256, 128, 64, 32, 16, 8]
parser.add_argument('--hidden_size', type=int, default=512, help='Number of units in hidden layer 1.')
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--embed_size', type=int, default=50, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-dir', type=str, default='./dataset/20ng_statis.pkl', help='type of dataset.')
parser.add_argument('--output-dir', type=str, default='torch_phi_output_etm_hier_share', help='type of dataset.')
parser.add_argument('--save-path', type=str, default='saves/gbn_model', help='type of dataset.')
parser.add_argument('--word-vector-path', type=str, default='../process_data/20ng_word_embedding.pkl', help='type of dataset.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_dataset = SetsDatasetText(args.dataset_dir, split='train', sample_size=args.train_sample_size)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                               num_workers=0, drop_last=True)

test_dataset = SetsDatasetText(args.dataset_dir, split='test', sample_size=args.test_sample_size)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=0, drop_last=True)

with open(args.dataset_dir, 'rb') as f:
        data = pickle.load(f)
args.vocab_size = 2000
voc = data['voc2000']

trainer = GBN_trainer(args,  voc_path=voc)
trainer.train(train_loader, test_loader)

trainer.vis_txt()
