from extract_features import tensor_from_ssm, labels_from_label_array
from util.load_data import load_ssm_string, load_segment_borders
from time import time
from sklearn.utils import shuffle

import argparse
import math
import numpy as np
import tensorflow as tf


def tdiff(timestamp: float) -> float:
    return time() - timestamp


def k(value: int) -> float:
    return float(value) / 1000


def feed(training_data, batch_size):
    X, Y = training_data
    X, Y = shuffle(X, Y)
    size = Y.shape[0]

    pointer = 0

    while pointer+batch_size < size:
        yield X[pointer:pointer+batch_size], Y[pointer:pointer+batch_size]
        pointer += batch_size
    yield X[pointer:], Y[pointer:]


def main(args):
    print("Starting training with parameters:", vars(args))

    # Load the data
    print("Loading data...")
    timestamp = time()
    ssm_string_data = load_ssm_string(args.data)
    segment_borders = load_segment_borders(args.data)
    print("Done in %.1fs" % tdiff(timestamp))

    # Figure out the maximum ssm size
    print("Gathering dataset statistics...")
    timestamp = time()
    max_ssm_size = 0
    counter = 0
    for ssm_obj in ssm_string_data.itertuples():
        counter += 1
        max_ssm_size = max(max_ssm_size, ssm_obj.ssm.shape[0])
    print("Done in %.1fs (%.2fk items, max ssm size: %d)" % (tdiff(timestamp), k(counter), max_ssm_size))

    # Producing training set
    train_buckets = dict()
    print("Producing training set...")
    counter = 0
    filtered = 0
    timestamp = time()
    for borders_obj in segment_borders.itertuples():
        counter += 1
        ssm = ssm_string_data.loc[borders_obj.id].ssm
        ssm_size = ssm.shape[0]
        if counter % 10000 == 0:
            print("  processed %.0fk items (%.1fs, filtered: %.1fk)" % (k(counter), tdiff(timestamp), k(filtered)))
            timestamp = time()
        if ssm_size < args.min_ssm_size or ssm_size > args.max_ssm_size:
            filtered += 1
            continue

        bucket_id = int(math.ceil(math.log2(ssm_size)))
        ssm_tensor = tensor_from_ssm(ssm, 2**bucket_id, args.window_size)
        ssm_labels = labels_from_label_array(borders_obj.borders, ssm_size)

        if bucket_id not in train_buckets:
            train_buckets[bucket_id] = ([], [])
        X, Y = train_buckets[bucket_id]
        X.append(ssm_tensor)
        Y.append(ssm_labels)
    del ssm_string_data
    del segment_borders

    # Compacting buckets and printing statistics
    print("Dataset buckets:")
    for bucket_id in train_buckets:
        X, Y = train_buckets[bucket_id]
        train_buckets[bucket_id] = (np.vstack(X), np.concatenate(Y))
        print("  max: %d len: %d" % (2**bucket_id, train_buckets[bucket_id][1].shape[0]))

    # Define the neural network



    # Training loop
    for epoch in range(args.max_epoch):
        for bucket_id in train_buckets:
            for batch_X, batch_Y in feed(train_buckets[bucket_id], args.batch_size):
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the lyrics segmentation cnn')
    parser.add_argument('--data', required=True,
                        help='The directory with the data')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--window-size', type=int, default=2,
                        help='The size of the window')
    parser.add_argument('--min-ssm-size', type=int, default=5,
                        help='Minimum size of the ssm matrix')
    parser.add_argument('--max-ssm-size', type=int, default=128,
                        help='Maximum size of the ssm matrix')

    args = parser.parse_args()
    main(args)
