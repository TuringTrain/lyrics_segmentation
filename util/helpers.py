from time import time
from sklearn.utils import shuffle
import numpy as np
import math


def compact_buckets(buckets: dict()) -> dict():
    """
    Compacts buckets (puts data inside in a big numpy array) and prints bucket statistics

    :param buckets: buckets of data of different sizes
    :return: compacted buckets
    """
    largest_bucket_id_len = (0, 0)
    for bucket_id in buckets:
        X, X_added, Y = buckets[bucket_id]
        #print('\n\n', X, X_added, Y)
        buckets[bucket_id] = (np.vstack(X), np.vstack(X_added), np.concatenate(Y))
        bucket_len = buckets[bucket_id][2].shape[0]
        print("  max: %3d len: %d" % (2 ** bucket_id, bucket_len))
        largest_bucket_len = largest_bucket_id_len[1]
        if bucket_len > largest_bucket_len:
            largest_bucket_id_len = (bucket_id, bucket_len)

    # Quick fix until I figure out how to process different sized buckets
    largest_bucket_id = largest_bucket_id_len[0]
    largest_bucket_content = buckets[largest_bucket_id]
    buckets = dict()
    buckets[largest_bucket_id] = largest_bucket_content
    return buckets


def window_features(feat_vector: np.ndarray) -> np.ndarray:
    """
        Given column vector of feature for line i, produce matrix indicating features in lines i-1, i, i+1
        note this is slightly bugged at first and last line

    :param feat_vector:
    :return:
    """
    return np.concatenate(
       (
           np.roll(feat_vector, -1, axis=0),
           feat_vector,
           np.roll(feat_vector, +1, axis=0)
       ),
       axis=1
    )


def feed_joint(data: (np.ndarray, np.ndarray, np.ndarray), ssm_size: int, batch_size: int, enable_shuffle=True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Produce random batches of data from the dataset

    :param data: tuple (X, X_added, Y) with feature for convolution X, feature for after convolution X_added, and labels Y
    :param batch_size: size of the batches
    :return: batch
    """
    X, X_added, Y = data
    if enable_shuffle:
        X, X_added, Y = shuffle(X, X_added, Y)
    size = len(Y)

    def put(X_batch, X_added_batch, X_lengths, Y_batch, i, X, X_added, Y):
        item = X[i]
        y = Y[i]
        added = X_added[i]
        pad_size = ssm_size - item.shape[0]

        X_lengths.append(item.shape[0])
        Y_batch.append(np.concatenate((y, np.zeros([pad_size]))))
        X_added_batch.append(np.concatenate((
            added,
            np.zeros([pad_size, added.shape[1]])  # Padding from right
        ), axis=0))
        X_batch.append(np.concatenate((
            item,
            np.zeros([pad_size, item.shape[1], item.shape[2], item.shape[3]])  # Padding from right
        ), axis=0))  # Column-wise

    pointer = 0
    while pointer < size:
        X_batch = []
        X_added_batch = []
        X_lengths = []
        Y_batch = []
        for i in range(pointer, min(pointer+batch_size, size)):
            put(X_batch, X_added_batch, X_lengths, Y_batch, i, X, X_added, Y)

        yield np.stack(X_batch), np.stack(X_added_batch), np.stack(X_lengths), np.stack(Y_batch)
        pointer += batch_size


def feed(data: (np.ndarray, np.ndarray, np.ndarray), batch_size: int, enable_shuffle=True) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Produce random batches of data from the dataset

    :param data: tuple (X, X_added, Y) with feature for convolution X, feature for after convolution X_added, and labels Y
    :param batch_size: size of the batches
    :return: batch
    """
    X, X_added, Y = data
    if enable_shuffle:
        X, X_added, Y = shuffle(X, X_added, Y)
    size = len(Y)

    pointer = 0
    while pointer+batch_size < size:
        yield np.concatenate(X[pointer:pointer+batch_size]), np.concatenate(X_added[pointer:pointer+batch_size]), np.concatenate(Y[pointer:pointer+batch_size])
        pointer += batch_size
    yield np.concatenate(X[pointer:]), np.concatenate(X_added[pointer:]), np.concatenate(Y[pointer:])


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


def windowdiff(seg1, seg2, k=None, boundary=1):
    """
    Compute the windowdiff score for a pair of segmentations.  A segmentation is any sequence
    over a vocabulary of two items (e.g. 0, 1, where the specified boundary value is used
    to mark the edge of a segmentation)
    If k is None it is half of an average segment length considering seg1 as true segments
    """

    assert len(seg1) == len(seg2), "Segments have unequal length: %d and %d" % (len(seg1), len(seg2))
    if k is None:
        k = max(1, round(len(seg1) / (2 * np.count_nonzero(seg1 == boundary))))
    assert k < len(seg1), "k (%d) can't be larger than a segment length (%d)" % (k, len(seg1))

    wd = 0
    for i in range(len(seg1) - k):
        wd += abs(np.count_nonzero(seg1[i:i+k+1] == boundary) - np.count_nonzero(seg2[i:i+k+1] == boundary))
    return wd / (len(seg1) - k)