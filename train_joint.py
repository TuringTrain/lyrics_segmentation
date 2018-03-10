from tensorflow.core.protobuf import config_pb2

from cnn.dense import Dense
from cnn.joint_rnn import JointRNN
from cnn.mnist_like import MnistLike
from cnn.no_padding_1conv import NoPadding1Conv
from extract_features import tensor_from_multiple_ssms, labels_from_label_array
from util.helpers import precision, recall, f1, k, tdiff, feed, compact_buckets, feed_joint

from util.load_data import load_ssm_string, load_ssm_phonetics, load_linewise_feature
from util.load_data import load_segment_borders, load_segment_borders_watanabe, load_segment_borders_for_genre

import argparse
import math
import numpy as np
import tensorflow as tf
from random import random
from time import time
from sklearn.metrics import confusion_matrix
from os import path
from tensorflow.contrib import slim


def add_to_buckets(buckets: dict(), bucket_id: int, tensor: np.ndarray, added_features: np.ndarray, labels: np.ndarray) -> None:
    if bucket_id not in buckets:
        buckets[bucket_id] = ([], [], [])
    X, X_added, Y = buckets[bucket_id]
    X.append(tensor)
    X_added.append(added_features)
    Y.append(labels)


def main(args):
    print("Starting training with parameters:", vars(args))

    # Load the data
    print("Loading data...")
    timestamp = time()

    # load different aligned SSMs
    multiple_ssms_data = [load_ssm_string(args.data),
                          #load_ssm_phonetics(args.data),
                          ]
    channels = len(multiple_ssms_data)
    print("Found", channels, "SSM channels")

    segment_borders = load_segment_borders(args.data)
    token_count_feat = load_linewise_feature(args.data, 'token_count')

    if not args.genre:
        train_borders, dev_borders, test_borders = load_segment_borders_watanabe(args.data)
    else:
        # load dev/test for some genre only (training is always on whole Watanabe train set)
        train_borders, dev_borders, test_borders = load_segment_borders_for_genre(args.data, args.genre)

    train_borders_set = set(train_borders.id)
    dev_borders_set = set(dev_borders.id)
    train_dev_borders_set = train_borders_set.union(dev_borders_set)
    print("Done in %.1fs" % tdiff(timestamp))

    # Figure out the maximum ssm size
    print("Gathering dataset statistics...")
    timestamp = time()
    max_ssm_size = 0
    counter = 0
    for ssm_obj in multiple_ssms_data[0].itertuples():
        current_id = ssm_obj.id
        # skip ids not in training or dev
        if not current_id in train_dev_borders_set:
            continue

        counter += 1
        max_ssm_size = max(max_ssm_size, ssm_obj.ssm.shape[0])
    print("Done in %.1fs (%.2fk items, max ssm size: %d)" % (tdiff(timestamp), k(counter), max_ssm_size))

    # Producing training set
    train_buckets = dict()
    test_buckets = dict()
    print("Producing training set...")
    counter = 0
    filtered = 0
    timestamp = time()
    max_ssm_size = min(max_ssm_size, args.max_ssm_size)

    # allow indexed access to dataframes
    for elem in multiple_ssms_data:
        elem.set_index(['id'], inplace=True)
    token_count_feat.set_index(['id'], inplace=True)

    for borders_obj in segment_borders.itertuples():
        counter += 1

        # temp. speedup for debugging
        #if counter % 100 != 0:
        #    continue

        current_id = borders_obj.id

        #skip ids not in training or dev
        if not current_id in train_dev_borders_set:
            continue

        ssm_elems = []
        for single_ssm in multiple_ssms_data:
            ssm_elems.append(single_ssm.loc[current_id].ssm)

        ssm_size = ssm_elems[0].shape[0]

        # Reporting
        if counter % 10000 == 0:
            print("  processed %3.0fk items (%4.1fs, filt.: %4.1fk)" % (k(counter), tdiff(timestamp), k(filtered)))
            timestamp = time()

        # Filter out too small or too large ssm
        if ssm_size < args.min_ssm_size or ssm_size > args.max_ssm_size:
            filtered += 1
            continue

        # Sentences are grouped into buckets to improve performance
        bucket_size = ssm_size
        if not args.buckets:
            bucket_size = max_ssm_size
        bucket_id = int(math.ceil(math.log2(bucket_size)))

        # one tensor for one song
        ssm_tensor = tensor_from_multiple_ssms(ssm_elems, 2**bucket_id, args.window_size)
        # concatenate all added features at axis=1 here
        added_features = token_count_feat.loc[current_id].feat_val
        added_feats_count = added_features.shape[1]

        ssm_labels = labels_from_label_array(borders_obj.borders, ssm_size)

        # fill train/test buckets according to definition files
        if current_id in train_borders_set:
            add_to_buckets(train_buckets, bucket_id, ssm_tensor, added_features, ssm_labels)
        else:
            assert current_id in dev_borders_set, 'id ' + current_id + ' is neither in train nor in dev!'
            add_to_buckets(test_buckets, bucket_id, ssm_tensor, added_features, ssm_labels)

    del multiple_ssms_data
    del added_features
    del segment_borders
    del train_borders
    del dev_borders
    del test_borders
    del train_borders_set
    del dev_borders_set
    del train_dev_borders_set

    # Compacting buckets and printing statistics
    #print("Training set buckets:")
    #train_buckets = compact_buckets(train_buckets)
    #print("Test set buckets:")
    #test_buckets = compact_buckets(test_buckets)

    # Define the neural network
    nn = JointRNN(window_size=args.window_size, ssm_size=2 ** next(train_buckets.keys().__iter__()), added_features_size=added_feats_count, channels=channels)

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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.device('/device:GPU:0'):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Checkpoint restore / variable initialising
            save_path = path.join(args.output, 'model.ckpt')
            latest_checkpoint = tf.train.latest_checkpoint(args.output)
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
                    for batch_X, batch_X_lengths, batch_Y in feed_joint(train_buckets[bucket_id], 2 ** next(train_buckets.keys().__iter__()), args.batch_size):
                        # Single training step
                        summary_v, global_step_v, loss_v, _ = sess.run(
                            fetches=[g_summary, g_global_step, nn.g_loss, g_train_op],
                            feed_dict={nn.g_in: batch_X,
                                       nn.g_labels: batch_Y,
                                       nn.g_dprob: 0.6,
                                       nn.g_lengths: batch_X_lengths})
                        summary_writer.add_summary(summary=summary_v, global_step=global_step_v)
                        avg_loss += loss_v

                        # Reporting
                        if global_step_v % args.report_period == 0:
                            print("Iter %d" % global_step_v)
                            print("  epoch %.0f, avg.loss %.4f, iter/s %.4fs" % (
                                epoch, avg_loss / args.report_period, tdiff(timestamp) / args.report_period
                            ))
                            timestamp = time()
                            avg_loss = 0.0

                        # Evaluation
                        if global_step_v % (args.report_period*10) == 0:
                            tp = 0
                            fp = 0
                            fn = 0
                            for bucket_id in test_buckets:
                                for test_X, test_X_lengths, true_Y in feed_joint(test_buckets[bucket_id], 2 ** next(train_buckets.keys().__iter__()), args.batch_size):
                                    # batch_size x max_len x 2
                                    pred_Y = nn.g_out.eval(feed_dict={
                                        nn.g_in: test_X,
                                        nn.g_dprob: 1.0,
                                        nn.g_lengths: test_X_lengths
                                    })
                                    for i in range(pred_Y.shape[0]):
                                        pred_sample_Y = np.argmax(pred_Y[i, :test_X_lengths[i], :], axis=1)
                                        true_sample_Y = true_Y[i, :test_X_lengths[i]]
                                        try:
                                            _, cur_fp, cur_fn, cur_tp = confusion_matrix(true_sample_Y, pred_sample_Y).ravel()
                                            tp += cur_tp
                                            fp += cur_fp
                                            fn += cur_fn
                                        except Exception as e:
                                            print(e)
                                            print(confusion_matrix(true_Y, pred_Y).ravel())
                                            print(confusion_matrix(true_Y, pred_Y))
                            print("  P: %.2f%%, R: %.2f%%, F1: %.2f%%" % (
                                precision(tp, fp) * 100, recall(tp, fn) * 100, f1(tp, fp, fn) * 100
                            ))

                        # Checkpointing
                        if global_step_v % 10000 == 0:
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
    parser.add_argument('--genre')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--window-size', type=int, default=2,
                        help='The size of the window')
    parser.add_argument('--min-ssm-size', type=int, default=5,
                        help='Minimum size of the ssm matrix')
    parser.add_argument('--max-ssm-size', type=int, default=128,
                        help='Maximum size of the ssm matrix')
    parser.add_argument('--report-period', type=int, default=100,
                        help='When to report stats')
    parser.add_argument('--buckets', default=False, action='store_true',
                        help='Enable buckets')

    args = parser.parse_args()
    main(args)
