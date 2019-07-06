
def model_fit(features, params, mode):
    
    """ multilayer perceptron (MLP)  """
    
    config = params
    
    input_layer = tf.reshape(features["x"], [-1, features["x"].shape[1] ] )
    print ('feature x shape', features["x"].shape)
    print ('reshape shape:', input_layer.shape)
    #trans = tf.string_to_number(input_layer)

    # Dense Layers
    #-------------------------------------------------------------
    hidden1 = tf.layers.dense(inputs=features["x"], units=config['n_hidden1'], activation=tf.nn.relu)
    
    training = tf.placeholder_with_default(False, shape=(), name='training')
    
    bn1 = tf.layers.batch_normalization(hidden1, momentum = 0.9)

    drop_h1 = tf.layers.dropout(inputs=bn1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    hidden2 = tf.layers.dense(inputs=drop_h1, units=config['n_hidden2'], activation=tf.nn.relu) 
    
    drop_h2 = tf.layers.dropout(inputs=hidden2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    hidden3 = tf.layers.dense(inputs=drop_h2, units=config['n_hidden3'], activation=tf.nn.relu)
    
    drop_h3 = tf.layers.dropout(inputs=hidden3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=drop_h3, units=config['nclasses'], activation=tf.nn.sigmoid)
    return logits
