from tensorflow as tf
# from tensorflow import 


class deepConvLSTM():
    def __init__(self, num_classes, input_shape, rnn_size=128, num_rnn_layers=2,
                 filter_size=[128, 256, 256], kernel_size=[3,3,3],
                 pool_size=[2,2,2], num_cnn_layers=3, dropout_rate=0.2):
        self.num_classes = num_classes
        self.input_data_shape = input_shape
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_rate = dropout_rate
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        # self._create_model()

    def _create_model(x):

        c1 = self.conv2d(x, self.kernel_size[0], self.filter_size[0], stride = 1, name = "conv1")
        m1 = self.maxpool2d(c1, k = self.pool_size[0])
        c2 = self.conv2d(m1, self.kernel_size[1], self.filter_size[1], stride = 1, name = "conv2")
        m2 = self.maxpool2d(c2, k = self.pool_size[0])
        c3 = self.conv2d(m2, self.kernel_size[2], self.filter_size[2], stride = 1, , name = "conv3")
        m3 = self.maxpool2d(c3, k = self.pool_size[0])
        ft = self.flatten(m3)
        fc1 = self.dense(ft,rnn_size, activation= 'relu', name = "dense1")
        d = tf.nn.dropout(fc1, rate = self.dropout_rate)
        out = self.dense(d, self.num_classes, activation= 'softmax', name = "dense2")

    # Create some wrappers for simplicity
    def conv1d(x, kernel_size, filter_size, strides=1, name = None):
        w = tf.get_variable(name+"/w",[kernel_size, 1, x.shape(2), filter_size], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b = tf.get_variable(name+"/b",filter_size,  initializer = tf.zeros_initializer())
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, w, strides=1 , padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Create some wrappers for simplicity
    def conv2d(x, kernel_size, filter_size, strides=1, name = None):
        w = tf.get_variable(name+"/w",,[kernel_size, 1, x.shape(2), filter_size], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b = tf.get_variable(name+"/b",,filter_size,  initializer = tf.zeros_initializer())
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')

    def flatten(x):
        return tf.reshape(x,[-1, x.shape(1) * x.shape(2) * x.shape(3)])

    def dense(x, output_size, name = None, activation = 'relu'):
        w = tf.get_variable(name+"/w",[x.shape(1), output_size], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b = tf.get_variable(name+"/b",, output_size,  initializer = tf.zeros_initializer())
        x = tf.add(tf.matmul(x, w), b)
        if activation = 'relu':
            return tf.nn.relu(x)
    



