from cnn.mnist_like import MnistLike
from extract_features import tensor_from_ssm, labels_from_label_array
from util.load_data import load_ssm_string, load_segment_borders

import argparse
import math
import numpy as np
import tensorflow as tf
from time import time
from sklearn.utils import shuffle
from os import path
from tensorflow.contrib import slim


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

        # Reporting
        if counter % 10000 == 0:
            print("  processed %3.0fk items (%4.1fs, filt.: %4.1fk)" % (k(counter), tdiff(timestamp), k(filtered)))
            timestamp = time()

        # Filter out too small or too large ssm
        if ssm_size < args.min_ssm_size or ssm_size > args.max_ssm_size:
            filtered += 1
            continue

        # Sentences are grouped into buckets to improve performance
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
    largest_bucket = (0, 0)
    for bucket_id in train_buckets:
        X, Y = train_buckets[bucket_id]
        train_buckets[bucket_id] = (np.vstack(X), np.concatenate(Y))
        bucket_len = train_buckets[bucket_id][1].shape[0]
        print("  max: %3d len: %d" % (2**bucket_id, bucket_len))
        if bucket_len > largest_bucket[1]:
            largest_bucket = (bucket_id, bucket_len)

    # Quick fix until I figure out how to process different sized buckets
    largest_bucket_content = train_buckets[largest_bucket[0]]
    train_buckets = dict()
    train_buckets[largest_bucket[0]] = largest_bucket_content

    # Define the neural network
    nn = MnistLike(window_size=args.window_size, ssm_size=2**largest_bucket[0])

    # Defining optimisation problem
    g_global_step = tf.train.get_or_create_global_step()
    g_train_op = slim.optimize_loss(
        loss=nn.g_loss, global_step=g_global_step, learning_rate=None,
        optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)

    # Logging
    summary_writer = tf.summary.FileWriter(
        logdir=path.join(args.output, 'log'), graph=tf.get_default_graph())
    g_summary = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        # Checkpoint restore / variable initialising
        checkpoint_path = path.join(args.output, 'checkpoint')
        save_path = path.join(checkpoint_path, 'model.ckpt')
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint is None:
            print("Initializing variables")
            timestamp = time()
            tf.get_variable_scope().set_initializer(tf.random_normal_initializer(mean=0.0, stddev=0.01))
            tf.global_variables_initializer().run()
            print("Done in %.2fs" % tdiff(timestamp))
        else:
            print("Restoring from checkpoint variables")
            timestamp = time()
            saver.restore(sess=sess, save_path=latest_checkpoint)
            print("Done in %.2fs" % tdiff(timestamp))

        print()
        timestamp = time()
        global_step_v = 0
        avg_loss = 0.0

        # Training loop
        for epoch in range(args.max_epoch):
            for bucket_id in train_buckets:
                for batch_X, batch_Y in feed(train_buckets[bucket_id], args.batch_size):
                    # Single training step
                    summary_v, global_step_v, loss_v, _ = sess.run(
                        fetches=[g_summary, g_global_step, nn.g_loss, g_train_op],
                        feed_dict={nn.g_in: batch_X, nn.g_labels: batch_Y, nn.g_dprob: 0.6})
                    summary_writer.add_summary(summary=summary_v, global_step=global_step_v)
                    avg_loss += loss_v

                    # Reporting
                    if global_step_v % args.report_period == 0:
                        print("iter %d, epoch %.0f, avg.loss %.2f, time per iter %.2fs" % (
                            global_step_v, epoch, avg_loss / args.report_period, tdiff(timestamp) / args.report_period
                        ))
                        timestamp = time()
                        avg_loss = 0.0

                    # Checkpointing
                    if global_step_v % 1000 == 0:
                        real_save_path = saver.save(sess=sess, save_path=save_path, global_step=global_step_v)
                        print("Saved the checkpoint to: %s" % real_save_path)

        real_save_path = saver.save(sess=sess, save_path=save_path, global_step=global_step_v)
        print("Saved the checkpoint to: %s" % real_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the lyrics segmentation cnn')
    parser.add_argument('--data', required=True,
                        help='The directory with the data')
    parser.add_argument('--output', required=True,
                        help='Output path')
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
    parser.add_argument('--report-period', type=int, default=1000,
                        help='When to report stats')

    args = parser.parse_args()
    main(args)
