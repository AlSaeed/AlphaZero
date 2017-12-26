import tensorflow as tf


class Net(object):
    def __init__(self, FILTERS=16, RES_BLOCKS=5, REG_FACTOR=0.0001, MOMENTUM_PARAM=0.9):
        self.FILTERS = FILTERS
        self.RES_BLOCKS = RES_BLOCKS
        self.REG_FACTOR = REG_FACTOR
        self.MOMENTUM_PARAM = MOMENTUM_PARAM

    def __batchNorm(self, X):
        mean, var = tf.nn.moments(X, axes=[0, 1, 2])
        return tf.nn.batch_normalization(X, mean, var, None, None, 0.001)

    def __fullyConnected(self, X, inSize, outSize):
        W = tf.Variable(tf.random_normal([inSize, outSize]), name='weights')
        B = tf.Variable(tf.constant(0.1), name='biases')
        return tf.matmul(X, W) + B

    def __convBlock(self, X):
        with tf.variable_scope('conv'):
            V = tf.Variable(tf.random_normal([3, 3, 3, self.FILTERS]), name='weights')
            C = tf.nn.conv2d(X, V, [1, 1, 1, 1], padding="SAME")
            N = self.__batchNorm(C)
        return tf.nn.relu(N)

    def __resBlock(self, X):
        with tf.variable_scope('conv1'):
            V1 = tf.Variable(tf.random_normal([3, 3, self.FILTERS, self.FILTERS]), name='weights')
            C1 = tf.nn.conv2d(X, V1, [1, 1, 1, 1], padding="SAME")
            N1 = self.__batchNorm(C1)
            R1 = tf.nn.relu(N1)
        with tf.variable_scope('conv2'):
            V2 = tf.Variable(tf.random_normal([3, 3, self.FILTERS, self.FILTERS]), name='weights')
            C2 = tf.nn.conv2d(R1, V2, [1, 1, 1, 1], padding="SAME")
            N2 = self.__batchNorm(C2)
        S = N2 + X
        return tf.nn.relu(S)

    def __policyHead(self, X):
        with tf.variable_scope('conv'):
            V = tf.Variable(tf.random_normal([1, 1, self.FILTERS, 2]), name='weights')
            C = tf.nn.conv2d(X, V, [1, 1, 1, 1], padding="SAME")
            N = self.__batchNorm(C)
            R = tf.reshape(tf.nn.relu(N), [-1, 18])
        with tf.variable_scope('fully_connected'):
            L = tf.reshape(self.__fullyConnected(R, 18, 9), [-1, 3, 3])
        return L

    def __valueHead(self, X):
        with tf.variable_scope('conv'):
            V = tf.Variable(tf.random_normal([1, 1, self.FILTERS, 1]), name='weights')
            C = tf.nn.conv2d(X, V, [1, 1, 1, 1], padding="SAME")
            N = self.__batchNorm(C)
            R1 = tf.reshape(tf.nn.relu(N), [-1, 9])
        with tf.variable_scope('fully_connected1'):
            FC = self.__fullyConnected(R1, 9, 64)
            R2 = tf.nn.relu(FC)
        with tf.variable_scope('fully_connected2'):
            rst = tf.nn.tanh(self.__fullyConnected(R2, 64, 1))
        return rst

    def inference(self, states):
        with tf.variable_scope('conv_block'):
            X = self.__convBlock(states)
        for i in range(1, self.RES_BLOCKS + 1):
            with tf.variable_scope('res_block' + str(i)):
                X = self.__resBlock(X)
        with tf.variable_scope('policy_head'):
            PH = self.__policyHead(X)
        with tf.variable_scope('value_head'):
            VH = self.__valueHead(X)
        return (PH, VH)

    def loss(self, p_logits, w, pi, z):
        reg = 0
        for v in tf.trainable_variables():
            reg += tf.nn.l2_loss(v)
        pi = tf.reshape(pi, [-1, 9])
        p_logits = tf.reshape(p_logits, [-1, 9])
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p_logits) + tf.square(w - z)
        return tf.reduce_mean(loss + self.REG_FACTOR * reg)

    def training(self, loss, learning_rate):
        optimizer = tf.train.MomentumOptimizer(learning_rate, self.MOMENTUM_PARAM)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def __str__(self):
        return 'Tic Tac Toe (FILTERS: ' + str(self.FILTERS) + ', RES_BLOCKS: ' + str(
            self.RES_BLOCKS) + ', REG_FACTOR: ' + str(self.REG_FACTOR) + ', MOMENTUM_PARAM: ' + str(
            self.MOMENTUM_PARAM) + ')'
