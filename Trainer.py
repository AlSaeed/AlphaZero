import tensorflow as tf
import numpy as np
from SelfPlay import SelfPlay
import os
import shutil


class Trainer(object):
    def __init__(self, experiment_name, game, network, c_puct, number_of_minibatches, learning_rate_policy_string,
                 mini_batch_size, games_between_versions, minibatches_between_versions, snapshot_period,
                 selection_lookback_span, independent_simulation_policy, rollouts_per_move, dirichlet_parameter,
                 epsilon, mcst_mini_batch_size, maximum_simulation_depth, tao_function_string):
        self._C_PUCT = c_puct
        self._INDEPENDENT_SIMULATION_POLICY = independent_simulation_policy
        self._ROLLOUTS_PER_MOVE = rollouts_per_move
        self._DIRICHLET_PARAMETER = dirichlet_parameter
        self._EPSILON = epsilon
        self._MCST_MINIBATCH_SIZE = mcst_mini_batch_size
        self._MAXIMUM_SIMULATION_DEPTH = maximum_simulation_depth
        self._TAO_FUNCTION_STRING = tao_function_string
        self._EXPERIMENT_NAME = experiment_name
        self._GAME = game
        self._NETWORK = network
        self._NUMBER_OF_MINIBATCHES = number_of_minibatches
        self._MINI_BATCH_SIZE = mini_batch_size
        self._GAMES_BETWEEN_VERSIONS = games_between_versions
        self._MINIBATCHES_BETWEEN_VERSIONS = minibatches_between_versions
        self._SNAPSHOT_PERIOD = snapshot_period
        self._SELECTION_LOOKBACK_SPAN = selection_lookback_span
        self._LEARNING_RATE_POLICY_STRING = learning_rate_policy_string
        self._SELF_PLAY = SelfPlay(game, c_puct, independent_simulation_policy, rollouts_per_move, dirichlet_parameter,
                                   epsilon, mcst_mini_batch_size, maximum_simulation_depth, eval(tao_function_string))
        self._initialize_network()

    def _initialize_network(self):
        actionsShape = self._GAME.actionsShape()
        actionsNumber = np.prod(actionsShape)
        self._STATES_INPUT = tf.placeholder(tf.float32, (None,) + self._GAME.stateShape())
        self._POLICY_INPUT = tf.placeholder(tf.float32, (None,) + actionsShape)
        self._VALUE_INPUT = tf.placeholder(tf.float32, (None, 1))
        self._LEARNING_RATE_INPUT = tf.placeholder(tf.float32, ())

        self._POLICY_HEAD, self._VALUE_HEAD = self._NETWORK.inference(self._STATES_INPUT)
        self._POLICY = tf.reshape(tf.nn.softmax(tf.reshape(self._POLICY_HEAD, [-1, actionsNumber])),
                                  (-1,) + actionsShape)
        self._LOSS = self._NETWORK.loss(self._POLICY_HEAD, self._VALUE_HEAD, self._POLICY_INPUT, self._VALUE_INPUT)
        self._TRAINING_STEP = self._NETWORK.training(self._LOSS, self._LEARNING_RATE_INPUT)

    def _pickMiniBatches(self, games):
        allData = []
        for g in games:
            allData += g
        cs = np.random.choice(len(allData), [self._MINIBATCHES_BETWEEN_VERSIONS, self._MINI_BATCH_SIZE])
        states = np.ndarray((self._MINIBATCHES_BETWEEN_VERSIONS, self._MINI_BATCH_SIZE,) + self._GAME.stateShape(),
                            np.float32)
        policies = np.ndarray((self._MINIBATCHES_BETWEEN_VERSIONS, self._MINI_BATCH_SIZE,) + self._GAME.actionsShape(),
                              np.float32)
        values = np.ndarray((self._MINIBATCHES_BETWEEN_VERSIONS, self._MINI_BATCH_SIZE, 1), np.float32)
        for m in range(self._MINIBATCHES_BETWEEN_VERSIONS):
            for i in range(self._MINI_BATCH_SIZE):
                c = cs[m][i]
                states[m][i], policies[m][i], values[m][i] = allData[c]
        return states, policies, values

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        learningRatePolicy = eval(self._LEARNING_RATE_POLICY_STRING)

        def inferenceFunction(states):
            P, W = sess.run([self._POLICY, self._VALUE_HEAD], {self._STATES_INPUT: states})
            return [(P[_], W[_]) for _ in range(len(P))]

        directory = os.path.join(os.getcwd(), "experiments", self._EXPERIMENT_NAME)
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        description = str(self)
        print description
        with open(directory + '/description', 'w') as f:
            f.write(description)

        saver = tf.train.Saver(max_to_keep=None)
        miniBatchNumber = 0
        games = []
        expLoss = 0
        while miniBatchNumber < self._NUMBER_OF_MINIBATCHES:
            games += self._SELF_PLAY.play(inferenceFunction, self._GAMES_BETWEEN_VERSIONS)
            if len(games) > self._SELECTION_LOOKBACK_SPAN:
                games = games[-self._SELECTION_LOOKBACK_SPAN:]
            states, policies, values = self._pickMiniBatches(games)
            for i in range(self._MINIBATCHES_BETWEEN_VERSIONS):
                lr = learningRatePolicy(miniBatchNumber)
                loss, _ = sess.run([self._LOSS, self._TRAINING_STEP],
                                   {self._STATES_INPUT: states[i], self._POLICY_INPUT: policies[i],
                                    self._VALUE_INPUT: values[i],
                                    self._LEARNING_RATE_INPUT: lr})
                miniBatchNumber += 1
                expLoss = 0.01 * loss + 0.99 * expLoss
                if miniBatchNumber % self._SNAPSHOT_PERIOD == 0:
                    print "Mini Batch: %7d, Loss: %5.3f" % (miniBatchNumber, expLoss / (1 - 0.99 ** miniBatchNumber))
                    saver.save(sess, directory + "/" + str(miniBatchNumber) + ".ckpt")
                if miniBatchNumber >= self._NUMBER_OF_MINIBATCHES:
                    break
        if miniBatchNumber % self._SNAPSHOT_PERIOD != 0:
            print "Mini Batch: %7d, Loss: %5.3f" % (miniBatchNumber, expLoss / (1 - 0.99 ** miniBatchNumber))
            saver.save(sess, directory + "/" + str(miniBatchNumber) + ".ckpt")

    def __str__(self):
        lines = [
            "Experiment: " + self._EXPERIMENT_NAME,
            "Game: " + str(self._GAME),
            "Network: " + str(self._NETWORK),
            "Number of Minibatches: " + str(self._NUMBER_OF_MINIBATCHES),
            "Minibatch Size: " + str(self._MINI_BATCH_SIZE),
            "Learning Rate Policy: " + self._LEARNING_RATE_POLICY_STRING,
            "Games between Versions: " + str(self._GAMES_BETWEEN_VERSIONS),
            "Minibatches between Versions: "+str(self._MINIBATCHES_BETWEEN_VERSIONS),
            "Snapshot Period: " + str(self._SNAPSHOT_PERIOD),
            "Selection Lookback Span: " + str(self._SELECTION_LOOKBACK_SPAN),
            "Tao Function: " + self._TAO_FUNCTION_STRING,
            "Independent Simulation Policy: " + str(self._INDEPENDENT_SIMULATION_POLICY),
            "Rollouts per Move: " + str(self._ROLLOUTS_PER_MOVE),
            "MCST Minibatch Size: " + str(self._MCST_MINIBATCH_SIZE),
            "Maximum Simulation Depth: " + str(self._MAXIMUM_SIMULATION_DEPTH),
            "C_PUCT: " + str(self._C_PUCT),
            "Epsilon: " + str(self._EPSILON),
            "Dirichlet Parameter: " + str(self._DIRICHLET_PARAMETER)
        ]
        return '\n'.join(lines)
