def cnn_model_fn(features, params, mode):  
  
"""Model function for CNN."""  
  config = params
  
  # Input Layer  
  
  input_layer = tf.reshape(tf.cast(features["x"], tf.float32), [-1, 154, 100, 2])
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=32, kernel_size=[1, 5], padding="same", activation=tf.nn.relu)

  # Pooling Layer #1
  
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=[1,2])

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d( inputs=pool1,filters=8,kernel_size=[1, 5], padding="same", activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 5], strides=[1,5])
  
  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(inputs=pool2,filters=2,kernel_size=[154, 5], padding="same", activation=tf.nn.relu)

  # Pooling Layer #3
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=[1, 2])

  # Dense Layer
  pool3_flat = tf.reshape(pool3, [-1, 154 * 5 * 2])
  dense = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer

  logits = tf.layers.dense(inputs=dropout, units=154)

  return logits
