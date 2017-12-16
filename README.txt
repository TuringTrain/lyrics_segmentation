Feature extraction:
From a square NxN picture (array with entries 0 <= x <= 1), windows
with specified radius (default 2) are cut. Result is a window radius * 2 + 1
height window centered around center.

CNN architecture:
Let's first start with the MNIST architecture. If it can handle digits,
it's probably more than enough to detect squares and diagonals.
(On the other hand there might be other pattern I'm not thinking of)
It has 2 convolutional (5x5) and two pooling layers (2x2).
Probably the 5x5 has to be adapted with the feature extraction window radius?

The SSMs are in the hdf files (id, ssm) and the segment borders are in
the segment borders dataframe (id, borders). Borders = [0, 13, 14] means e.g.
that lines 0, 13 and 14 are labeled as text segment ending (y=1). As one-hot
encoding this gives a vector of length = line count of corresponding song
= ssm[song_id].shape[0] where positions 0, 13 and 14 are 1s, otherwise 0.

Training test splits I generate with learning_tools.train_val_test_split(.).
It only gives u the train and val, so u cannot mess accidentally with the test. ;)
(I will test in on test set in the end, once we have settled on the concrete models)

Your first goal would be to use only string encoded SSMs in order to predict
labels using the simple MNIST architecture. Related work gets 57% by using
only SSMs on the same data set (however not same split, I don't know their split).
They feature engineer similar to convulutions (see Watanabe paper).
