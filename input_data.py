import pickle
import numpy as np


class DataSet:
    def one_hot(self, labels):
        new_label = []
        for label in labels:
            row = []
            for i in range(10):
                if i == label:
                    row.append(1)
                else:
                    row.append(0)
            new_label.append(row)
        return np.array(new_label, ndmin=2)


class TrainSet(DataSet):
    def __init__(self, data, labels, one_hot=False):
        self.images = np.array(data, ndmin=2)
        if one_hot:
            self.labels = super().one_hot(labels)
        else:
            self.labels = labels

        self.num_examples = len(data)
        self.__index = 0

    def next_batch(self, batch_size):
        x, y = self.images[self.__index: self.__index + batch_size], self.labels[self.__index: self.__index + batch_size]
        self.__index += batch_size
        return x, y


class TestSet(DataSet):
    def __init__(self, data, labels, one_hot=False):
        self.images = np.array(data, ndmin=2)
        if one_hot:
            self.labels = super().one_hot(labels)
        else:
            self.labels = labels


class Cifar:
    def __init__(self, batches, test, one_hot=False):
        data = []
        labels = []
        for batch in batches:
            for i in range(len(batch[b'data'])):
                data.append(batch[b'data'][i])
                labels.append(batch[b'labels'][i])
        self.train = TrainSet(data, labels, one_hot)
        self.test = TestSet(test[b'data'], test[b'labels'], one_hot)


def read_data_sets(url, one_hot=False):
    batches = []
    for i in range(1, 6):
        with open(url + 'data_batch_' + str(i), 'rb') as f:
            batches.append(pickle.load(f, encoding='bytes'))
            f.close()
    with open(url + 'test_batch', 'rb') as f:
        test = pickle.load(f, encoding='bytes')
        f.close()
    return Cifar(batches, test, one_hot)



