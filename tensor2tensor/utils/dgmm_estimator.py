import tensorflow as tf

class Estimator:
    def __init__(self, k, z, is_training=True):
        with tf.variable_scope('estimator'):
            self.input_tensor = z
            # TODO: change dimensionality
            # transformer input in here
            net = self.base_dense_layer(self.input_tensor, 32, name='dense_0', is_training=is_training)
            # net = base_dense(net, 256, name='dense_1', is_training=is_training)
            # net = base_dense(net, 128, name='dense_2', is_training=is_training)
            net = self.base_dense_layer(net, 64, name='dense_3', is_training=is_training)
            net = self.base_dense_layer(net, 32, name='dense_4', is_training=is_training)
            net = self.base_dense_layer(net, k, name='dense_5', is_training=is_training, bn=False)
            self.output_tensor = tf.nn.softmax(net, name='predicted_memebership')

    def base_dense_layer(self, input_layer, units, name='dense', is_training=True, bn=True, activation_fn=tf.nn.leaky_relu):
        with tf.variable_scope(name):
            net = tf.layers.dense(input_layer,
                                  units)
            if bn:
                net = tf.layers.batch_normalization(net, training=is_training, name='batch_normalization')

            if activation_fn is not None:
                net = activation_fn(net)

            return net