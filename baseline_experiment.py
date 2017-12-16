import numpy as np
import ssmfeatures
import learning_tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def line_feat_matrix_labelled(some_songs, ssm_lookup):
    #read first entry to find out the dimensionality of feature vector
    for song in some_songs.itertuples():
        line_feature_matrix = np.empty([0, song_line_features_labels_id(song, ssm_lookup).shape[1]], dtype=np.float64)
        break
    for song in some_songs.itertuples():
        line_feature_matrix = np.append(line_feature_matrix, song_line_features_labels_id(song, ssm_lookup), axis=0)
    return line_feature_matrix




def song_line_features_labels_id(song, ssm_lookup):
    #declared count of feats here to that appending arrays can work below
    #hope its possible without this ugly hack?
    #1 = number of SSMs per song (only string encoding = 1)
    #4 = number of Watanabe features per threshold
    #9 number of thresholds
    #3 features without thresholds, cf. line 53
    feat_count = 1 * (4 * 9 + 3)
    song_line_feature_matrix = np.empty([0, feat_count], dtype=np.float64)

    #song is a tuple from df.itertuples()
    song_id = song[0]
    ssm_lines_string = ssm_lookup.loc[song_id].ssm
    #ssm_lines_postag = sppm.loc[song_id].ssm

    #compute features for whole song
    feat0_for_threshold, feat1_for_threshold,\
    feat2_for_threshold, feat3_for_threshold,\
    frpf3, frpf4b, frpf4e                       = ssmfeatures.ssm_feats_thresholds_watanabe(ssm_lines_string)

    #feat4_for_threshold, feat5_for_threshold,\
    #feat6_for_threshold, feat7_for_threshold = ssm_feats_thresholds_watanabe(ssm_lines_postag)

    line_count = len(ssm_lines_string)
    for line_index in range(line_count):
        feats_of_line = []
        #compute features for several thresholds 0.1, ..., 0.9 (cf. Watanabe paper)
        for lam in [0.1 * factor for factor in range(1, 10)]:
            feats_of_line.append(feat0_for_threshold[lam].get(line_index, 0))
            feats_of_line.append(feat1_for_threshold[lam].get(line_index, 0))
            feats_of_line.append(feat2_for_threshold[lam].get(line_index, 0))
            feats_of_line.append(feat3_for_threshold[lam].get(line_index, 0))
            #feats_of_line.append(feat4_for_threshold[lam].get(line_index, 0))
            #feats_of_line.append(feat5_for_threshold[lam].get(line_index, 0))
            #feats_of_line.append(feat6_for_threshold[lam].get(line_index, 0))
            #feats_of_line.append(feat7_for_threshold[lam].get(line_index, 0))
        ##add non-thresholded features here
        feats_of_line.append(frpf3.get(line_index, 0))
        feats_of_line.append(frpf4b.get(line_index, 0))
        feats_of_line.append(frpf4e.get(line_index, 0))

        song_line_feature_matrix = np.append(song_line_feature_matrix, np.array([feats_of_line]), axis=0)

    #concatenate features, labels and id
    #segment_ending_indices = set(rd.segment_borders(song, 'mldb'))
    #instead of computing these indices, load it from mldb_seg5p_segment_borders.hdf

    song_line_labels = np.array([1 if line_index in segment_ending_indices else 0 for line_index in range(line_count)]).reshape(-1,1)

    song_line_id = np.array([song_id for line_index in range(line_count)]).reshape(-1,1)

    song_line_feature_matrix = np.concatenate((song_line_feature_matrix, song_line_labels, song_line_id), axis=1)
    return song_line_feature_matrix





def baseline_experiment():
    import time
    start_time = time.time()

    feat_count = 1 * (4 * 9 + 3)
    songs = load_corpus()
    sssm = load_ssm()

    songs_train, songs_val = learning_tools.train_val_test_split(songs, subset_factor=0.1)
    print('Training set size  :', len(songs_train))
    print('Validation set size:', len(songs_val))

    print()
    print('Computing feature vectors...')
    XY_train = line_feat_matrix_labelled(songs_train, sssm)
    XY_val = line_feat_matrix_labelled(songs_val, sssm)

    print('Got', XY_train.shape[0], 'training instances and', XY_train.shape[1] - 2, 'features')
    print()
    print('Scaling each feature for itself...')
    XY_train = learning_tools.scale_features(XY_train)
    XY_val = learning_tools.scale_features(XY_val)


    print('Separating features X, labels Y, and groups G (the group of a line is the song it belongs to)...')
    X_train = XY_train[:, :feat_count]
    Y_train = XY_train[:, feat_count].ravel()
    G_train = XY_train[:, feat_count + 1].ravel()

    X_val = XY_val[:, :feat_count]
    Y_val = XY_val[:, feat_count].ravel()
    G_val = XY_val[:, feat_count + 1].ravel()

    print('Fitting model on training set...')
    #Logit as they use in the paper...
    model = LogisticRegression(class_weight='balanced')

    model.fit(X_train, Y_train)

    print('Predicting labels on validation set...')
    model_prediction = model.predict(X_val)

    print()
    print(time.time() - start_time)
    print('Precision:', precision_score(y_true=Y_val, y_pred=model_prediction))
    print('Recall   :', recall_score(y_true=Y_val, y_pred=model_prediction))
    print('F-Score  :', f1_score(y_true=Y_val, y_pred=model_prediction))
