import numpy as np
import pickle

def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = len(labels_dense)
  labels_one_hot = []
  for i in range(len(labels_dense)):
      if labels_dense[i] == 0:
          labels_one_hot.append([1,0])
      elif labels_dense[i] == 1:
          labels_one_hot.append([0,1])
  return np.asarray(labels_one_hot)

class Dataset(object):
    def __init__(self,subbands,subints,DMs,labels,data_size,one_hot = True,is_fake_data = False):
        self._data_size = data_size
        self._subbands = subbands
        self._subints = subints
        self._DMs = DMs
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._samples_num = len(self._subbands)
        self._one_hot = one_hot
    @property
    def data_size(self):
        return self._data_size
    @property
    def subbands(self):
        return self._subbands
    @property
    def subints(self):
        return self._subints
    @property
    def DMs(self):
        return self._DMs
    @property
    def lables(self):
        return self._labels
    @property
    def epoch_completed(self):
        return self._epochs_completed
    @property
    def samples_num(self):
        return self._samples_num

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._samples_num:
            # finish epoch
            self._epochs_completed +=1

            # shuffle the data
            data_index = list(range(self._samples_num))
            np.random.shuffle(data_index)
            self._subbands = [self._subbands[i] for i in data_index]
            self._subints = [self._subints[i] for i in data_index]
            self._DMs = [self._DMs[i] for i in data_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._samples_num
        end = self._index_in_epoch
        if self._one_hot:
            labels_re = dense_to_one_hot(self._labels[start:end],2)
        return [self._subbands[start:end], self._subints[start:end], self._DMs[start:end],labels_re]
def get_sub_bands_ints_DMs(p_path, n_path):

    all_subbands = []
    all_subints = []
    all_DMs = []
    all_labels = []
    with open(p_path,'rb') as pklfile:
        print('正在读取：',p_path,'里面的数据文件...')
        p_subbands = pickle.load(pklfile, encoding='iso-8859-1')
        all_subbands.extend(p_subbands)
        p_subints = pickle.load(pklfile, encoding='iso-8859-1')
        all_subints.extend(p_subints)
        p_DMs = pickle.load(pklfile, encoding='iso-8859-1')
        all_DMs.extend(p_DMs)
        all_labels.extend([1]*len(p_subbands))
    with open(n_path, 'rb') as pklfile:
        print('正在读取：',n_path,'里面的数据文件...')
        n_subbands = pickle.load(pklfile, encoding='iso-8859-1')
        all_subbands.extend(n_subbands)
        n_subints = pickle.load(pklfile, encoding='iso-8859-1')
        all_subints.extend(n_subints)
        n_DMs = pickle.load(pklfile, encoding='iso-8859-1')
        all_DMs.extend(n_DMs)
        all_labels.extend([0]*len(n_subbands))
    # shuffle
    data_index = list(range(len(all_labels)))
    np.random.shuffle(data_index)
    all_subbands = [all_subbands[i] for i in data_index]
    all_subints = [all_subints[i] for i in data_index]
    all_DMs = [all_DMs[i] for i in data_index]
    all_labels = [all_labels[i] for i in data_index]
    f_len, p_len = all_subbands[0].shape
    t_len, p_len = all_subints[0].shape
    return [f_len,t_len,p_len],all_subbands, all_subints, all_DMs, all_labels
def read_data(Candidates_pos_path,Candidates_neg_path,test_part):
    class Datasets(object):
        pass
    datasets = Datasets()
    data_size,all_subbands,all_subints,all_DMs, all_labels = get_sub_bands_ints_DMs(Candidates_pos_path,Candidates_neg_path)
    Candidates_num = len(all_subbands)
    samples_index = list(range(Candidates_num))
    train_subbands = [all_subbands[i] for i in samples_index[:int(Candidates_num*(1-test_part))]]
    train_subints = [all_subints[i] for i in samples_index[:int(Candidates_num*(1-test_part))]]
    train_DMs = [all_DMs[i] for i in samples_index[:int(Candidates_num*(1-test_part))]]
    train_labels = [all_labels[i] for i in samples_index[:int(Candidates_num*(1-test_part))]]

    test_subbands = [all_subbands[i] for i in samples_index[int(Candidates_num*(1-test_part)):]]
    test_subints = [all_subints[i] for i in samples_index[int(Candidates_num*(1-test_part)):]]
    test_DMs = [all_DMs[i] for i in samples_index[int(Candidates_num*(1-test_part)):]]
    test_labels = [all_labels[i] for i in samples_index[int(Candidates_num*(1-test_part)):]]

    datasets.train = Dataset(train_subbands,train_subints,train_DMs,train_labels,data_size,is_fake_data=False)
    datasets.test = Dataset(test_subbands,test_subints,test_DMs,test_labels,data_size,is_fake_data=False)

    return datasets

if __name__ == '__main__':
    p_path = 'H:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309p_pfd_sub_band_int.pkl'
    n_path = 'H:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309n_pfd_sub_band_int.pkl'
    all_data = read_data(p_path, n_path,test_part=0.3)
    train_data = all_data.train
    print(train_data.next_batch(20)[3])
