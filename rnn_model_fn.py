import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
import tensorflow.contrib.rnn as rnn


def simple_rnn(features, labels, mode, params):
    """
    Recurrent Neural Networks
    """
    config = params
    
    # 0. Reformat input shape to become a sequence
    x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)
    #print 'x={}'.format(x)
    
    # 1. configure the RNN
    lstm_cell = rnn.BasicLSTMCell(conf['LSTM_SIZE'], forget_bias=1.0)
    outputs, _ = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # slice to keep only the last cell of the RNN
    outputs = outputs[-1]
    #print 'last outputs={}'.format(outputs)
  
    # output is result of linear activation of last layer of RNN
    weight = tf.Variable(tf.random_normal([conf['LSTM_SIZE'], N_OUTPUTS]))
    bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
    output = tf.matmul(outputs, weight) + bias
    return output
