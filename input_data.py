import pickle
import numpy as np


class DataSet:
    def __init__(self, data, labels, one_hot=False):
        self.images = np.array(data, ndmin=2)
        if one_hot:
            self.labels = self.__one_hot(labels)
        else:
            self.labels = labels

        self.num_examples = len(data)
        self.__index = 0

    @staticmethod
    def __one_hot(labels):
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

    def next_batch(self, batch_size):
        if self.__index + batch_size > self.images.shape[0]:
            self.__index = 0
        x, y = self.images[self.__index: self.__index + batch_size], self.labels[self.__index: self.__index + batch_size]
        self.__index = (self.__index + batch_size) % self.images.shape[0]
        return x, y

    def normalize(self):
        self.images = np.array(list(map(lambda x: x/255, self.images)))


class Cifar:
    def __init__(self, batches, test, one_hot=False):
        data = []
        labels = []
        for batch in batches:
            for i in range(len(batch[b'data'])):
                data.append(batch[b'data'][i])
                labels.append(batch[b'labels'][i])

        self.train = DataSet(data[:int(4 * len(data) / 5)], labels[:int(4 * len(data) / 5)], one_hot)
        self.validation = DataSet(data[int(4 * len(data) / 5):], labels[int(4 * len(data) / 5):], one_hot)
        self.test = DataSet(test[b'data'], test[b'labels'], one_hot)

    def normalize_data(self):
        self.train.normalize()
        self.validation.normalize()
        self.test.normalize()


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



