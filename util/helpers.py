from time import time
from sklearn.utils import shuffle
import numpy as np


def compact_buckets(buckets: dict()) -> dict():
    """
    Compacts buckets (puts data inside in a big numpy array) and prints bucket statistics

    :param buckets: buckets of data of different sizes
    :return: compacted buckets
    """
    largest_bucket = (0, 0)
    for bucket_id in buckets:
        X, Y = buckets[bucket_id]
        buckets[bucket_id] = (np.vstack(X), np.concatenate(Y))
        bucket_len = buckets[bucket_id][1].shape[0]
        print("  max: %3d len: %d" % (2 ** bucket_id, bucket_len))
        if bucket_len > largest_bucket[1]:
            largest_bucket = (bucket_id, bucket_len)

    # Quick fix until I figure out how to process different sized buckets
    largest_bucket_content = buckets[largest_bucket[0]]
    buckets = dict()
    buckets[largest_bucket[0]] = largest_bucket_content
    return buckets


def feed(data: (np.ndarray, np.ndarray), batch_size: int) -> (np.ndarray, np.ndarray):
    """
    Produce random batches of data from the dataset

    :param data: tuple (X, Y) with training samples and labels
    :param batch_size: size of the batches
    :return: batch
    """
    X, Y = data
    X, Y = shuffle(X, Y)
    size = Y.shape[0]

    pointer = 0
    while pointer+batch_size < size:
        yield X[pointer:pointer+batch_size], Y[pointer:pointer+batch_size]
        pointer += batch_size
    yield X[pointer:], Y[pointer:]


def tdiff(timestamp: float) -> float:
    """
    Compute time offset (for time reporting purposes)
    """
    return time() - timestamp


def k(value: int) -> float:
    """
    Shorthand for thousands
    """
    return float(value) / 1000


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def f1(tp: int, fp: int, fn: int) -> float:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return 2 * prec * rec / (prec + rec)