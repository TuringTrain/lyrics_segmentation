import argparse
import numpy as np

from util.evaluation import pr_approximate_randomization, pr_measures, pr_class_one
from os import listdir, path

from util.helpers import windowdiff


def read_file(filename):
    true = []
    pred = []
    true_batch = []
    pred_batch = []

    with open(filename, 'r') as reader:
        for line in reader:
            if line.startswith('='):
                if len(true_batch) > 0:
                    true.append(np.array(true_batch))
                    true_batch = []
                    pred.append(np.array(pred_batch))
                    pred_batch = []
                continue

            row = [int(ele) for ele in line.rstrip().split('\t')]
            true_batch.append(row[0])
            pred_batch.append(row[1])

    if len(true_batch) > 0:
        true.append(np.array(true_batch))
        pred.append(np.array(pred_batch))

    return true, pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the lyrics segmentation cnn')
    parser.add_argument('--input', required=True,
                        help='Input file')

    args = parser.parse_args()
    print("Starting evaluation with parameters:", vars(args))

    # Figuring out models
    models = {}
    evaluations = {}
    best = {}
    for model in listdir(args.input):
        if model.startswith("."):
            continue
        model_dir = path.join(args.input, model)
        if model not in models:
            models[model] = {}
            evaluations[model] = {}
        best_feature_set = None
        best_score = 0
        for feature_set in listdir(model_dir):
            if feature_set.startswith("."):
                continue
            orig_name = feature_set
            feature_set = feature_set[2:]
            loaded_file = read_file(path.join(path.join(model_dir, orig_name), "eval.txt"))
            evaluation = pr_class_one(np.concatenate(loaded_file[0]), np.concatenate(loaded_file[1]))

            if evaluation[2] > best_score:
                best_score = evaluation[2]
                best_feature_set = feature_set

            if feature_set not in evaluations[model] or evaluations[model][feature_set][2] < evaluation[2]:
                evaluations[model][feature_set] = evaluation
                models[model][feature_set] = loaded_file

            wd = 0
            wd_count = 0
            for i in range(len(loaded_file[0])):
                wd += windowdiff(loaded_file[0][i], loaded_file[1][i])
                wd_count += 1
            wd /= wd_count

            print("Run   %30s    p: %.4f" % (model+"@"+orig_name, evaluation[0]))
            print("      %30s    r: %.4f" % ("", evaluation[1]))
            print("      %30s   f1: %.4f" % ("", evaluation[2]))
            print("      %30s   wd: %.4f" % ("", wd))
        best[model] = best_feature_set

    for model in models:
        best_predictions = models[model][best[model]]
        best_eval = evaluations[model][best[model]]
        for feature_set in models[model]:
            cur_predictions = models[model][feature_set]
            cur_eval = evaluations[model][feature_set]
            significance = pr_approximate_randomization(
                np.concatenate(best_predictions[0]),
                np.concatenate(best_predictions[1]),
                np.concatenate(cur_predictions[1]),
                iterations=100000
            )
            outcome = "[NO] "
            if significance[2] < 0.05:
                outcome = "[YES]"
            print("Significance %s p: %.5f, r: %.5f, f1: %.5f" % (outcome, significance[0], significance[1], significance[2]))
            print("     %30s p: %.4f, r: %.4f, f1: %.4f" % (model+"@"+feature_set, cur_eval[0], cur_eval[1], cur_eval[2]))
            print("  vs %30s p: %.4f, r: %.4f, f1: %.4f" % (model+"@"+best[model], best_eval[0], best_eval[1], best_eval[2]))

