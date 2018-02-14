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


def tensor_from_two_ssms(ssm_some: np.ndarray, ssm_other: np.ndarray, pad_to_size: int, half_window=2) -> np.ndarray:
    """
    Produce a tensor containing all line-wise features for multiple given similarity matrix

    :param ssm_some: square similarity matrix
    :param ssm_other: square similarity matrix
    :param pad_to_size: pad feature matrices to a certain size (maximum size of the image in the dataset)
    :param half_window: the size of the window
    :return: |ssm size| feature matrices of dimensionality: pad_to_size (width) x 2*half_window (height) x 2 (channels)
    """
    # SSMs have to be square
    assert ssm_some.shape[0] == ssm_some.shape[1]
    assert ssm_other.shape[0] == ssm_other.shape[1]

    # SSMs have to be of same size
    assert ssm_some.shape[0] == ssm_other.shape[0]

    ssm_size = ssm_some.shape[0]

    # dimensions of the final tensor
    dim_x = ssm_size
    dim_layers = 2    # number of SSM layers used
    dim_y = 2 * half_window
    dim_z = pad_to_size
    tensor = np.empty([dim_x, dim_layers, dim_y, dim_z], dtype=np.float32)
    for line in range(ssm_size):
        # lower and upper bounds of the window
        lower = line - half_window + 1
        upper = line + half_window + 1

        # padding size for left-right padding
        pad_lr_size = float(pad_to_size - ssm_size) / 2

        # pad SSM 1
        # pad top and down
        td_padded_patch_some = np.concatenate((
            np.zeros([max(-lower, 0), ssm_size], dtype=np.float32),  # Padding from top
            ssm_some[max(lower, 0):min(upper, ssm_size), :],
            np.zeros([max(upper - ssm_size, 0), ssm_size])  # Padding from bottom
        ), axis=0)  # Row-wise

        # pad left and right
        td_lr_padded_patch_some = np.concatenate((
            np.zeros([dim_y, int(math.floor(pad_lr_size))]),  # Padding from left
            td_padded_patch_some,
            np.zeros([dim_y, int(math.ceil(pad_lr_size))])  # Padding from right
        ), axis=1)  # Column-wise

        # pad SSM 2
        # pad top and down
        td_padded_patch_other = np.concatenate((
            np.zeros([max(-lower, 0), ssm_size], dtype=np.float32),  # Padding from top
            ssm_other[max(lower, 0):min(upper, ssm_size), :],
            np.zeros([max(upper - ssm_size, 0), ssm_size])  # Padding from bottom
        ), axis=0)  # Row-wise

        td_lr_padded_patch_other = np.concatenate((
            np.zeros([dim_y, int(math.floor(pad_lr_size))]),  # Padding from left
            td_padded_patch_other,
            np.zeros([dim_y, int(math.ceil(pad_lr_size))])  # Padding from right
        ), axis=1)  # Column-wise


        # stack SSMs on top of each other
        td_lr_padded_patch_layered = np.stack([td_lr_padded_patch_some, td_lr_padded_patch_other], axis=0)

        tensor[line] = td_lr_padded_patch_layered

        # print('\n\n\n\n\n\n\n\n')
        # print('ssm_size:', ssm_size)
        # print()
        # print('np.zeros([max(-lower, 0), ssm_size], dtype=np.float32).shape:', np.zeros([max(-lower, 0), ssm_size], dtype=np.float32).shape)
        # print('ssm_some[max(lower, 0):min(upper, ssm_size), :].shape:', ssm_some[max(lower, 0):min(upper, ssm_size), :].shape)
        # print('np.zeros([max(upper - ssm_size, 0), ssm_size]).shape:', np.zeros([max(upper - ssm_size, 0), ssm_size]).shape)
        # print('td_padded_patch_some.shape:', td_padded_patch_some.shape)
        # print()
        # print('np.zeros([dim_y, int(math.floor(pad_lr_size))]).shape:', np.zeros([dim_y, int(math.floor(pad_lr_size))]).shape)
        # print('td_padded_patch_some.shape:', td_padded_patch_some.shape)
        # print('np.zeros([dim_y, int(math.ceil(pad_lr_size))]).shape:', np.zeros([dim_y, int(math.ceil(pad_lr_size))]).shape)
        # print('td_lr_padded_patch_some.shape:', td_lr_padded_patch_some.shape)
        # print()
        # print('np.zeros([max(-lower, 0), ssm_size], dtype=np.float32).shape:', np.zeros([max(-lower, 0), ssm_size], dtype=np.float32).shape)
        # print('ssm_other[max(lower, 0):min(upper, ssm_size), :].shape:', ssm_some[max(lower, 0):min(upper, ssm_size), :].shape)
        # print('np.zeros([max(upper - ssm_size, 0), ssm_size]).shape:', np.zeros([max(upper - ssm_size, 0), ssm_size]).shape)
        # print('td_padded_patch_other.shape:', td_padded_patch_other.shape)
        # print()
        # print('np.zeros([dim_y, int(math.floor(pad_lr_size))]).shape:', np.zeros([dim_y, int(math.floor(pad_lr_size))]).shape)
        # print('td_padded_patch_other.shape:', td_padded_patch_other.shape)
        # print('np.zeros([dim_y, int(math.ceil(pad_lr_size))]).shape:', np.zeros([dim_y, int(math.ceil(pad_lr_size))]).shape)
        # print('td_lr_padded_patch_other.shape:', td_lr_padded_patch_other.shape)
        # print()
        # print('td_lr_padded_patch_layered.shape', td_lr_padded_patch_layered.shape)
        # print()
        # print('tensor.shape:', tensor.shape)
    tensor = np.swapaxes(tensor, 1, 2)
    tensor = np.swapaxes(tensor, 2, 3)
    return tensor


def remove_main_diagonal(ssm: np.ndarray):
    assert ssm.shape[0] == ssm.shape[1]
    ssm_size = ssm.shape[0]

    ssm_nodiag = np.empty([ssm_size, ssm_size - 1])
    for line in range(ssm_size):
        line_nodiag = np.concatenate((ssm[line, 0:line], ssm[line, line+1:ssm_size]), axis=0)
        ssm_nodiag[line] = line_nodiag
    return ssm_nodiag

def tensor_from_two_ssms_removed_diagonals(ssm_some: np.ndarray, ssm_other: np.ndarray, pad_to_size: int, half_window=2) -> np.ndarray:
    """
    Produce a tensor containing all line-wise features for multiple given similarity matrix

    :param ssm_some: square similarity matrix
    :param ssm_other: square similarity matrix
    :param pad_to_size: pad feature matrices to a certain size (maximum size of the image in the dataset)
    :param half_window: the size of the window
    :return: |ssm size| feature matrices of dimensionality: pad_to_size (width) x 2*half_window (height) x 2 (channels)
    """

    ssm_some = remove_main_diagonal(ssm_some)
    ssm_other = remove_main_diagonal(ssm_other)

    # SSMs have to be squarish
    assert ssm_some.shape[0] == ssm_some.shape[1] + 1
    assert ssm_other.shape[0] == ssm_other.shape[1] + 1

    # SSMs have to be of same size
    assert ssm_some.shape[0] == ssm_other.shape[0]
    assert ssm_some.shape[1] == ssm_other.shape[1]

    ssm_height = ssm_some.shape[0]
    ssm_width = ssm_some.shape[1]

    # dimensions of the final tensor
    dim_x = ssm_height
    dim_layers = 2    # number of SSM layers used
    dim_y = 2 * half_window
    dim_z = pad_to_size
    tensor = np.empty([dim_x, dim_layers, dim_y, dim_z], dtype=np.float32)
    for line in range(ssm_height):
        # lower and upper bounds of the window
        lower = line - half_window + 1
        upper = line + half_window + 1

        # padding size for left-right padding
        pad_lr_size = float(pad_to_size - ssm_width) / 2

        # pad SSM 1
        # pad top and down
        td_padded_patch_some = np.concatenate((
            np.zeros([max(-lower, 0), ssm_width], dtype=np.float32),  # Padding from top
            ssm_some[max(lower, 0):min(upper, ssm_height), :],
            np.zeros([max(upper - ssm_height, 0), ssm_width])  # Padding from bottom
        ), axis=0)  # Row-wise

        # pad left and right
        td_lr_padded_patch_some = np.concatenate((
            np.zeros([dim_y, int(math.floor(pad_lr_size))]),  # Padding from left
            td_padded_patch_some,
            np.zeros([dim_y, int(math.ceil(pad_lr_size))])  # Padding from right
        ), axis=1)  # Column-wise

        # pad SSM 2
        # pad top and down
        td_padded_patch_other = np.concatenate((
            np.zeros([max(-lower, 0), ssm_width], dtype=np.float32),  # Padding from top
            ssm_other[max(lower, 0):min(upper, ssm_height), :],
            np.zeros([max(upper - ssm_height, 0), ssm_width])  # Padding from bottom
        ), axis=0)  # Row-wise

        td_lr_padded_patch_other = np.concatenate((
            np.zeros([dim_y, int(math.floor(pad_lr_size))]),  # Padding from left
            td_padded_patch_other,
            np.zeros([dim_y, int(math.ceil(pad_lr_size))])  # Padding from right
        ), axis=1)  # Column-wise


        # stack SSMs on top of each other
        td_lr_padded_patch_layered = np.stack([td_lr_padded_patch_some, td_lr_padded_patch_other], axis=0)

        tensor[line] = td_lr_padded_patch_layered

    tensor = np.swapaxes(tensor, 1, 2)
    tensor = np.swapaxes(tensor, 2, 3)
    return tensor
