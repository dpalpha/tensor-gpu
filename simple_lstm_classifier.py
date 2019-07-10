import tensorflow as tf


def lstm_rnn__model_fn(features, labels, mode):

    X = tf.expand_dims(features, axis=2)

    batchsize = tf.shape(X)[0]

    seqlen = tf.shape(X)[1]

    cells = [tf.nn.rnn_cell.GRUCell(RNN_CELLSIZE) for _ in range(N_LAYERS)]

    cells[:-1] = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=DROPOUT_PKEEP) for cell in cells[:-1]]

    # a stacked RNN cell still works like an RNN cell
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)

    # X[BATCHSIZE, SEQLEN, 1], Hin[BATCHSIZE, RNN_CELLSIZE*N_LAYERS]
    Yn, H = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    Yn = tf.reshape(Yn, [batchsize * seqlen, RNN_CELLSIZE])
    logit = tf.layers.dense(Yn, units=2, activation=tf.nn.sigmoid)  # logit [BATCHSIZE*SEQLEN, 1]
    logit = tf.reshape(logit, [batchsize, seqlen, 2])  # logit [BATCHSIZE, SEQLEN, 1]
    return logit