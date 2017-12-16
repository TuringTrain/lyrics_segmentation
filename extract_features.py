import numpy as np

#get line-wise windows from a picture by horizontally slicing it
#the feature of a line is the partial picture that is centered around the line index
#e.g. if half window size is 2, the window is 5 including its center
#feature of line 4 would be the picture with lines [line 2, line 3, line 4, line 5, line 6]
#picture is a numpy.ndarray

#3D tensor containing all line-wise features (=sub pictures)
def tensor_from_picture(picture, half_window=2):
    dim_x = picture.shape[0]
    dim_y = 2 * half_window + 1
    dim_z= picture.shape[1]
    tensor = np.empty([dim_x, dim_y, dim_z], dtype=np.float32)
    for line_index in range(picture.shape[0]):
        subpic = subpicture_from(picture, line_index, half_window)
        tensor[line_index] = subpic
    return tensor

def subpicture_from(picture, line_index, half_window):
    subpicture = np.empty([0, picture.shape[1]], dtype=np.float32)
    for index in range(line_index - half_window, line_index + half_window + 1):
        if index < 0 or index >= picture.shape[0]:
            #padding with zeros
            current_line = np.zeros(picture.shape[1]).reshape(1,-1)
        else:
            current_line = picture[index, :].reshape(1,-1)
        subpicture = np.append(subpicture, current_line, axis=0)
    return subpicture
