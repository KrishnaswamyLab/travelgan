import numpy as np

class Loader(object):
    def __init__(self, data, labels=None, shuffle=False):
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.labels_given = labels is not None

        if shuffle:
            self.r = list(range(data.shape[0]))
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=100):
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start + batch_size] for x in self.data]
            self.start += batch_size
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0] - self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1, x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows - self.start)

        if not self.labels_given:  # don't return length-1 list
            return batch[0]
        else:  # return list of data and labels
            return batch

    def iter_batches(self, batch_size=100):
        num_rows = self.data[0].shape[0]

        start = 0
        end = batch_size

        for i in range(num_rows // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            if not self.labels_given:
                yield [x[start:end] for x in self.data][0]
            else:
                yield [x[start:end] for x in self.data]

        if batch_size > num_rows:
            if not self.labels_given:
                yield [x for x in self.data][0]
            else:
                yield [x for x in self.data]
        if end != num_rows:
            if not self.labels_given:
                yield [x[end:] for x in self.data][0]
            else:
                yield [x[end:] for x in self.data]
























