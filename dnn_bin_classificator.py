
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


feat_cols= []
for col in x_train.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))

x_train = x_train.apply(lambda x: (x-x.mean())/x.std())
x_val = x_val.apply(lambda x: (x-x.mean())/x.std())

my_head = tf.contrib.estimator.binary_classification_head()

nn_model = tf.estimator.DNNClassifier(
    hidden_units=[336,168,84,42],
    feature_columns=feat_cols,
    activation_fn=tf.nn.relu,
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=0.01
    ),
    n_classes=2,
    batch_norm = True
    #                                        optimizer=tf.train.AdagradOptimizer(learning_rate=0.06),
    #model_dir="/tmp/kaggle_fraud",
    #config=chk_point_run_config,
)
def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], \
            num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }

nn_model = tf.estimator.add_metrics(nn_model, metric_auc)

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_tr,batch_size=100, num_epochs=1000, shuffle=True)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_val,y=y_val,batch_size=100,num_epochs=1, shuffle=False)

validation_log_losses = []

for i in range(0,100):
    
    nn_model.train(input_fn= input_func, steps=100)

    nn_model.evaluate(eval_input_func)

    predict_input_func = tf.estimator.inputs.pandas_input_fn(x=x_val,batch_size=100,num_epochs=1,shuffle=False)

    predictions = nn_model.predict(input_fn=predict_input_func)

    prediction=[i['probabilities'][1] for i in predictions]
    
    validation_log_loss = metrics.log_loss(y_val, prediction)
    
    print("moment %02d : %0.2f" % (i, training_log_loss))
    
    validation_log_losses.append(validation_log_loss)
      print("Model training finished.")

    print("auc scores oos: %s "% roc_auc_score(y_val,prediction))

  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(validation_log_losses, label="validation")
  plt.legend()


