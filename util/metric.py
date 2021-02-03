from sklearn.metrics import accuracy_score
import numpy as np


class AccMetric:
    """
    Compute classification accuracy of current batch or averaged across the whole set.
    """
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def clear(self):
        """
        clear history of prediction.
        """
        self.y_true.clear()
        self.y_pred.clear()

    def record(self, y_true, y_pred, clear):
        """
        record a batch of prediction.
        :param y_true: N dimensional np array of ground-truth classes.
        :param y_pred: N dimensional np array of predicted classes.
        :param clear: bool. True: clear history; False: do not clear history.
        :return acc: average accuracy of current batch of prediction.
        """
        if clear:
            self.clear()
        else:
            self.y_true.append(y_true)
            self.y_pred.append(y_pred)

        return accuracy_score(y_true, y_pred)

    def average(self, clear):
        """
        compute average accuracy of history of prediction.
        :param clear: bool. True: clear history; False: do not clear history.
        :return acc: average accuracy of history of prediction.
        """
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)

        if clear: self.clear()

        return accuracy_score(y_true, y_pred)


def median_acc(acc_file, k=10):
    """
    return the median of the last k val acc from the visdom json file
    :param acc_file: the file that records all val acc history
    :param k: int. last k val acc to consider
    :return: float. median of acc
    """
    acc = []
    with open(acc_file) as f:
        for _, line in enumerate(f):
            acc.append(float(line.rstrip()))

    k = min(k, len(acc))

    return np.median(acc[-k:])
