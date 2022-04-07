
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
tf.compat.v1.reset_default_graph()
import librosa
tf.compat.v1.disable_eager_execution()

class RNN:
    def __init__(self):
        # Audio parameters
        self.audio_length = 2
        self.sampling_rate = 44100
        step = 512  # numbers of point librosa takes to create 20 features
        self.time_steps = self.audio_length * self.sampling_rate // step

        # RNN parameters
        self.learning_rate = 0.005
        self.display_step = 50
        self.test_step = 180
        self.nb_epochs = 1200
        self.batch_size = 74
        self.n_features = 100
        self.rnn_sizes = [128, 128]
        self.model_name = 'mfcc_cross'
        self.load = False

        self.export_dir = './networks/'
        self.import_dir = './input/'
        self.label_file = 'train.csv'
        self.n_classes=2


    def extract_mfcc(self, fl, train=False):
        """
        Split the audio files in segments of 2 seconds, and compute the mfcc
        features for all of the
        :param train: True for train set, False for test set
        """
        path = 'audio_test/'

        mfccs = []
        fnames = []
        count = 0

        for fn in glob.glob(
                os.path.join(self.import_dir + path, fl)):
            fname = fn.split('/')[-1]
            sound_clip, s = librosa.load(fn, sr=self.sampling_rate)

            try:
                mfcc = librosa.feature.mfcc(y=sound_clip, sr=s,
                                            n_mfcc=self.n_features).T
            except ValueError:
                # Some files of the test set cause problems
                mfcc = np.ones((10, self.n_features))
            time_steps, _ = mfcc.shape

            mfcc = scale(mfcc, axis=0)

            pad = self.time_steps - time_steps % self.time_steps

            # pad with zeros to have the same time_steps
            if pad < self.time_steps // 3 or time_steps // self.time_steps == 0:
                mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant',
                              constant_values=(0, 0))
                mfcc = mfcc.reshape(time_steps // self.time_steps + 1,
                                    self.time_steps, self.n_features)

            # remove the last part if it is too short
            else:
                mfcc = mfcc[:time_steps // self.time_steps * self.time_steps, :]
                mfcc = mfcc.reshape(time_steps // self.time_steps,
                                    self.time_steps,
                                    self.n_features)

            for i in range(mfcc.shape[0]):
                mfccs.append(mfcc[i, :, :])
                fnames.append(fname)

            count += 1
            if count % 100 == 0:
                print("file {}".format(count))
            
        df_mfcc = pd.DataFrame({'fname': fnames,})

        df_mfcc.to_csv(self.import_dir + 'mfcc_test.csv')
        with open('./input/mfcc_test.p', 'wb') as fp:
            pickle.dump(np.array(mfccs), fp)

    def build_rnn(self, x, keep_prob):
        """
        Build the network
        """
        layer = {
            'weight': tf.Variable(
                tf.compat.v1.truncated_normal([self.rnn_sizes[-1], self.n_classes],
                                    stddev=0.01)),
            'bias': tf.Variable(tf.constant(0.1, shape=[self.n_classes]))}
        lstm_cells = [rnn_cell.LSTMCell(rnn_size) for rnn_size in
                      self.rnn_sizes]
        drop_cells = [
            tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob) for
            lstm in lstm_cells]

        lstm = rnn_cell.MultiRNNCell(drop_cells)
        output, state = tf.compat.v1.nn.dynamic_rnn(lstm, x, dtype=tf.float32,
                                          sequence_length=self.length(x))
        last = self.last_relevant(output, self.length(x))

        return tf.nn.softmax(
            tf.tensordot(last, layer['weight'], [[1], [0]]) + layer[
                'bias'])

    @staticmethod
    def length(sequence):
        """
        From https://danijar.com/variable-sequence-lengths-in-tensorflow/
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        """
        Return the last relevant output of the LSTM cell, by removing the
        trailing zeros (!! Raise an error it the array is full of zeros)
        From https://danijar.com/variable-sequence-lengths-in-tensorflow/
        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def predict(self):
        with open('./input/mfcc_test.p', 'rb') as fp:
            X = pickle.load(fp)

        df_mfcc = pd.read_csv(self.import_dir + 'mfcc_test.csv')
        predictions = np.zeros((5, len(pd.unique(df_mfcc.fname)), self.n_classes))
        for j in range(5):
            # RNN
            tf.compat.v1.reset_default_graph()
            # variables
            x = tf.compat.v1.placeholder("float", [None, self.time_steps, self.n_features])
            keep_prob = tf.compat.v1.placeholder("float", name='keep_prob')

            prediction = self.build_rnn(x, keep_prob)
            with tf.compat.v1.Session() as session:
                saver = tf.compat.v1.train.Saver()
                saver.restore(session, self.export_dir + self.model_name + '{}'.format(j))
                unique = pd.unique(df_mfcc.fname)
                for i in range(len(pd.unique(df_mfcc.fname))):
                    idxs = df_mfcc.fname[df_mfcc.fname == unique[i]].index.tolist()
                    batch = X[idxs, :, :]
                    if batch.sum() == 0:
                        batch = np.ones_like(batch)
                    pred = session.run(prediction, feed_dict={x: np.array(batch), keep_prob: 1})
                    # Average on the segment for the same audio
                    predictions[j, i, :] = pred.mean(axis=0)

        unique = pd.unique(df_mfcc.fname)
        results = {'label': [], 'fname': []}
        # Average on the 5 networks
        predictions = predictions.mean(axis=0)
        for i in range(len(pd.unique(df_mfcc.fname))):
            top3_labels = self.top_3(predictions[i, :], return_string=False)
            return top3_labels[0]

    def top_3(self, predictions, return_string=True):
        top_labels = np.argsort(predictions)
        top_labels = top_labels[::-1]
        top3_labels = top_labels[:3]

        if return_string:
            top3_labels = " ".join([self.enc.inverse_transform(el) for el in top3_labels])
        return top3_labels


if __name__ == '__main__':
    rnn = RNN()
    rnn.extract_mfcc("068.wav", train=False)
    res=rnn.predict()
    print(res)