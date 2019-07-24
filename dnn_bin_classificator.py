
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc, roc_auc_score
    
x_train, x_val, y_tr, y_val =  model_selection.train_test_split(X_train, y_train,test_size=.5, random_state=1200)

tf.set_random_seed(47)
tf.reset_default_graph()

def sm_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)

x_train = x_train.apply(lambda x: (x-x.mean())/x.std())
x_val = x_val.apply(lambda x: (x-x.mean())/x.std())

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss

import tensorflow as tf
from tensorflow.python.client import device_lib
print("GPU Available: ", tf.test.is_gpu_available())
print(list(map(lambda x: x.name ,device_lib.list_local_devices())))



# tf.set_random_seed(47)
# tf.reset_default_graph()

x_train, x_val, y_tr, y_val =  model_selection.train_test_split(X_train, y_train,
                                                                test_size=.8, random_state=1200)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset

feature_columns = []

for key in x_train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])

# settings = tf.estimator.\
#                            RunConfig(keep_checkpoint_max=None, 
#                                      save_checkpoints_steps=None, 
#                                      save_checkpoints_secs=None,
#                                      train_distribute=strategy
#                                     ).\
#                            replace(
#                                       session_config=tf.ConfigProto(log_device_placement=True,
#                                                                     allow_soft_placement=True,
#                                                                     device_count={'GPU': 1, 'CPU': 1},
#                                                                     inter_op_parallelism_threads=1,
#                                                                     intra_op_parallelism_threads=8
#                                                                    )
# )

    
classifier = tf.estimator.DNNClassifier(
    
    feature_columns=feature_columns,
    
    hidden_units=[320, 128, 16],
    
    activation_fn=tf.nn.relu,
    
    dropout=0.03,
    
#     optimizer=tf.train.ProximalAdagradOptimizer(
#         learning_rate=0.05,
#         l1_regularization_strength=0.01
#     ),
    n_classes=2,
    batch_norm = False,
    #model_dir="/kag_fraud_sys

)

def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], \
            num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }

classifier = tf.estimator.add_metrics(classifier, metric_auc)

batch_size = 100
train_steps = 400

for i in range(0,100):
    classifier.train(input_fn=lambda:train_input_fn(x_train, y_tr,batch_size),steps=train_steps)
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(x_val, y_val,batch_size))
    predictions = classifier.predict(input_fn=lambda:eval_input_fn(x_val,labels=None,batch_size=batch_size))
    prediction=[i['probabilities'][1] for i in predictions]
    print("auc scores oos: %s "%(roc_auc_score(y_val,prediction)))

predictions = classifier.predict(input_fn=lambda:eval_input_fn(X_test,labels=None,batch_size=batch_size))
prediction=[i['probabilities'][1] for i in prediction
