import tensorflow as tf

def cifar_model_fn(features, mode):
    """Model function for cifar10"""
    # Input layer
    x = tf.reshape(features, [-1, 32, 32, 3])

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[
                             3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV1')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN1')

    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[
                            3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV2')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN2')
    
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[
                            3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV3')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN3')
    
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[
                             3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV4')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN4')
    
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[
                                    3, 3], strides=2, padding='same', name='POOL1')

    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=3, padding='same',
                             activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV5')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN5')
    
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[
                                    3, 3], strides=2, padding='same', name='POOL2')
    # Dense layer
    x = tf.reshape(x, [-1, 8 * 8 * 128])

    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='DENSE1')
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='DENSE2')
    x = tf.layers.dropout(inputs=x, rate=FLAGS.dropout_rate,
                                training=mode == tf.estimator.ModeKeys.TRAIN, name='DROPOUT')

    logits = tf.layers.dense(inputs=x, units=10,
                             kernel_regularizer=regularizer, name='FINAL')
