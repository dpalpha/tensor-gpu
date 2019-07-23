
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)


def generate_time_series(datalen):
    freq1 = 0.2
    freq2 = 0.15
    noise = [np.random.random() * 0.1 for i in range(datalen)]
    x1 = np.sin(np.arange(0, datalen) * freq1) + noise
    x2 = np.sin(np.arange(0, datalen) * freq2) + noise
    x = x1 + x2
    return x.astype(np.float32)


DATA_SEQ_LEN = 24000

data = generate_time_series(DATA_SEQ_LEN)

SEQLEN = 16  # unrolled sequence length
BATCHSIZE = 32

X = data
Y = np.roll(data, -1)

Y = Y>0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y.astype(int),
                                                    test_size=.2,
                                                    random_state=0)

X_train = np.reshape(X_train, [-1, SEQLEN])
Y_train = np.reshape(Y_train, [-1, SEQLEN])

X_test = np.reshape(X_test, [-1, SEQLEN])
Y_test = np.reshape(Y_test, [-1, SEQLEN])


def train_dataset():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(DATA_SEQ_LEN * 4 // SEQLEN)
    dataset = dataset.batch(BATCHSIZE)
    samples, labels = dataset.make_one_shot_iterator().get_next()
    return samples, labels


def eval_dataset():
    evaldataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    evaldataset = evaldataset.repeat(1)
    evaldataset = evaldataset.batch(BATCHSIZE)

    samples, labels = evaldataset.make_one_shot_iterator().get_next()
    return samples, labels


RNN_CELLSIZE = 80

N_LAYERS = 3

DROPOUT_PKEEP = 0.88

def model_rnn_fn(features, labels, mode):

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


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logit, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logit, name="softmax_tensor")
    }

    loss = train_op = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logit)
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_op = tf.contrib.training.create_train_op(loss, optimizer)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


if __name__ == '__main__':

    training_config = tf.estimator.RunConfig()

    estimator = tf.estimator.Estimator(model_fn=model_rnn_fn, config=training_config)

    estimator.train(input_fn=train_dataset,steps=2000)

    results = estimator.predict(eval_dataset)

    predict = [result["probabilities"] for result in results]
    predict = list(map(lambda x : x[0], np.array(predict)[:,-1]))

    actual = Y_test[:, -1]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams['figure.figsize'] = 15, 7

    plt.plot(actual, label="Actual Values", color='green')
    plt.plot(list(map(lambda x: 1 if x> 0.5 else 0, predict)),
             label="Predicted Values", color='red', )

    plt.show()
