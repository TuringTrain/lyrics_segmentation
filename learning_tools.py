from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import numpy as np

#fixed split with train, validation, test proportion of 60%-20%-20%
#return no test fold currently
def train_val_test_split(songs, train_part = 0.6, val_part = 0.2, test_part = 0.2, subset_factor=1):
    if subset_factor > 1 or subset_factor <= 0:
        raise ValueError('subset factor must be in (0,1]')
    if subset_factor == 1:
        songs_subset = songs
    else:
        songs_subset = train_test_split(songs, train_size=subset_factor, test_size=1-subset_factor, random_state=0)[0]

    train_size = 1 - test_part
    songs_train_val, songs_test = train_test_split(songs_subset, test_size = test_part, train_size = train_size, random_state=0)
    songs_train, songs_val = train_test_split(songs_train_val, train_size = train_part/train_size, random_state=0)
    return songs_train, songs_val


#Scale each feature for itself
#label_offset is 2 if right to the features is a label and a song id
def scale_features(line_feature_matrix, label_offset=2):
    line_feature_matrix = np.array(line_feature_matrix, dtype=np.float64)
    #the last column is the label, don't scale that :o)
    for column_index in range(line_feature_matrix.shape[1] - label_offset):
        #scaler needs column vector
        column_normalized = RobustScaler().fit_transform(line_feature_matrix[:, column_index].reshape(-1,1))
        column_normalized = MinMaxScaler().fit_transform(column_normalized)
        #transform back to row vector
        line_feature_matrix[:, column_index] = column_normalized.reshape(1,-1)
    return line_feature_matrix
