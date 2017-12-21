import numpy as np
import math
import json


def labels_from_label_array(label_array, ssm_size):
    if isinstance(label_array, str):
        label_array = json.loads(label_array)

    labels = np.zeros(ssm_size, dtype=np.int32)
    for label in label_array:
        # Workaround for buggy labels, probably not needed anymore
        # if label >= labels.shape[0]:
        #    continue
        labels[label] = 1
    return labels


def tensor_from_ssm(ssm: np.ndarray, pad_to_size: int, half_window=2) -> np.ndarray:
    """
    Produce a tensor containing all line-wise features for the given similarity matrix

        the feature of a line is the partial picture that is centered around the line index
        e.g. if half window size is 2, the window is 5 including its center
        feature of line 4 would be the picture with lines [line 2, line 3, line 4, line 5, line 6]

    :param ssm: square similarity matrix
    :param pad_to_size: pad feature matrices to a certain size (maximum size of the image in the dataset)
    :param half_window: the size of the window
    :return: n feature matrices of size pad_to_size x 2*half_window, where n equals to ssm size
    """
    assert ssm.shape[0] == ssm.shape[1] # SSM has to be square
    ssm_size = ssm.shape[0]

    # dimensions of the final tensor
    dim_x = ssm_size
    dim_y = 2 * half_window
    dim_z = pad_to_size
    tensor = np.empty([dim_x, dim_y, dim_z], dtype=np.float32)
    for line in range(ssm_size):
        # lower and upper bounds of the window
        lower = line - half_window + 1
        upper = line + half_window + 1

        unpadded_patch = np.concatenate((
            np.zeros([max(-lower, 0), ssm_size], dtype=np.float32),  # Padding from top
            ssm[max(lower, 0):min(upper, ssm_size), :],
            np.zeros([max(upper - ssm_size, 0), ssm_size])  # Padding from bottom
        ), axis=0)  # Row-wise

        pad_size = float(pad_to_size - ssm_size) / 2
        tensor[line] = np.concatenate((
            np.zeros([dim_y, int(math.floor(pad_size))]),  # Padding from left
            unpadded_patch,
            np.zeros([dim_y, int(math.ceil(pad_size))])  # Padding from right
        ), axis=1)  # Column-wise
    return tensor
