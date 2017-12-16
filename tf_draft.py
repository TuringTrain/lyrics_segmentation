import tensorflow as tf
import numpy as np

#4D tensor, containing all 3D tensors per picture, for all songs (=pictures)
moat = np.empty([len(mldb), 0, 0, 0], dtype=np.float32)
half_window_size = 2

# Define execution graph
tf_tensor = tf.placeholder(tf.float32, shape=(None, half_window_size*2+1, None))
fancy_calculation = tf.reduce_mean(tf_tensor, axis=[1, 2])


with tf.Session() as sess:
    # Initilising variables
    pass

    # Main whatever loop
    for song in mldb.itertuples():
        song_id = song[0]
        song_ssm = sssm_string.loc[song_id].ssm
        tensor = tensor_from_picture(song_ssm, half_window_size)
        output = sess.run(fancy_calculation, feed_dict={tf_tensor: tensor})
        print(output, len(output))
        #print(tensor[0])
        #print(moat[42, :, :, :].shape)
        #moat[42, :, :, :] = np.append(moat[42, :, :, :], tensor, axis=0)
        #print(rd.segment_borders(song, 'mldb'))
        break
