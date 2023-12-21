import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
import time
from torch.utils import data

class SetsDatasetText(data.Dataset):
    def __init__(self, data_dir, split='train', sample_size=3):
        self.sample_size = sample_size
        with open(data_dir, 'rb') as f:
            data = pickle.load(f)

        if split == 'train':
            self.data_input = data['train_class_data']
        elif split == 'test':
            self.data_input = data['test_class_data']

        self.N_class = len(self.data_input)

    def __getitem__(self, item):

        seed = int(str(time.time()).split('.')[1])
        np.random.seed(seed=seed)

        all_doc = self.data_input[item]
        choice_doc_index = np.random.choice(len(all_doc), size=self.sample_size, replace=False)

        return np.array([all_doc[doc_index] for doc_index in choice_doc_index]).astype("int32")

    def __len__(self):
        return self.N_class

