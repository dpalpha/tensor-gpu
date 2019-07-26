
from tensorflow.python.client import device_lib
import tensorflow as tf
import gc
device_lib.list_local_devices()

# mlp classifier settings 
#------------------------------------------------------------------------------------------------------
config = {}
config['model_dir'] = 'model8/'
config['nclasses'] = 2
config['LSTM_SIZE'] = 3  # number of hidden layers in each of the LSTM cells
config['BATCH_SIZE'] = 20
config['learning_rate']=0.02
config['NUM_EPOCHS'] = 200
config['NUM_STEPS'] = int(X_train.shape[0]/config['BATCH_SIZE'])
config['n_hidden1'] = 126
config['n_hidden2'] = 64
config['n_hidden3'] = 32
config['dropout_rate'] = 0.05


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

settings = tf.estimator.\
                           RunConfig(keep_checkpoint_max=2, 
                                     save_checkpoints_steps=config['NUM_STEPS'], 
                                     save_checkpoints_secs=None,
                                     train_distribute=strategy).\
                           replace(
                                      session_config=tf.ConfigProto(log_device_placement=True,
                                      device_count={'GPU': 1}))



        
def mlp_model_fn(features, params, mode):
    
    config = params
    
    input_layer = tf.reshape(features["x"], [-1, features["x"].shape[1] ] )
    print ('feature x shape', features["x"].shape)
    print ('reshape shape:', input_layer.shape)
    #trans = tf.string_to_number(input_layer)

    # Dense Layers
    #-------------------------------------------------------------
    hidden1 = tf.layers.dense(inputs=features["x"], units=config['n_hidden1'], activation=tf.nn.relu)
        
    bn1 = tf.layers.batch_normalization(hidden1, momentum = 0.9)

    drop_h1 = tf.layers.dropout(inputs=bn1, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    hidden2 = tf.layers.dense(inputs=drop_h1, units=config['n_hidden2'], activation=tf.nn.relu) 
    
    drop_h2 = tf.layers.dropout(inputs=hidden2, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    hidden3 = tf.layers.dense(inputs=drop_h2, units=config['n_hidden3'], activation=tf.nn.relu)
    
    drop_h3 = tf.layers.dropout(inputs=hidden3, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=drop_h3, units=config['nclasses'], activation=tf.nn.sigmoid)
    
    return logits


def model_tffn(features, labels, mode, params):
    
    """Model function for MLP."""

    config = params

    # Input Layer
    
    logits = mlp_model_fn(features,params, mode)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=config['nclasses'])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=config['learning_rate'])
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step(), name="minimieze")
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)


    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Create the Estimator
mlp_classifier = tf.estimator.Estimator(
model_fn=model_tffn, model_dir=config['model_dir'],
    params=config)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=100)

summary_hook = tf.train.SummarySaverHook(
    save_steps=100,
    output_dir='./tmp/rnnStats',
    scaffold=tf.train.Scaffold(),
    summary_op=tf.summary.merge_all())


# Train the model
#----------------------------------------------------------
train_input_fn = tf.estimator.\
inputs.numpy_input_fn(
  x={"x": X_train},
  y=y_train,
  batch_size=config['BATCH_SIZE'],
  num_epochs=config['NUM_EPOCHS'],
  num_threads=3,
  shuffle=True,
)

# Evaluate the model and print results
#----------------------------------------------------------
eval_input_fn = tf.estimator.\
inputs.numpy_input_fn(
x={"x": X_val},
y=y_val,
num_epochs=1,
shuffle=False)

# run train
mlp_classifier.\
train(
  input_fn=train_input_fn,
  steps=config['NUM_STEPS'],
  hooks=[logging_hook])


#hooks=[ValidationHook(estimator, validation_input_fn, None, STEPS_PER_EPOCH)]

eval_results = mlp_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

predictions_eval = list(mlp_classifier.predict(input_fn=eval_input_fn))
predicted_proba_eval = list(map(lambda x: list(x)[1], [p["probabilities"] for p in predictions_eval]))


# Predict the model
#----------------------------------------------------------
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
# x={"x": X_test.values},
# num_epochs=1,
# shuffle=False)

# predictions = list(mlp_classifier.predict(input_fn=test_input_fn))
# predicted_proba = list(map(lambda x: list(x)[1], [p["probabilities"] for p in predictions]))
