import numpy as np
import math


class _Node(object):
    def __init__(self, state, tree, noise=False):
        self._GAME = tree.GAME
        self._STATE = state
        self._P = self._GAME.player(state)
        self._DIRICHLET_PARAMETER = tree.DIRICHLET_PARAMETER
        self._EPSILON = tree.EPSILON
        self._C_PUCT = tree.C_PUCT
        self._ACTIONS_SHAPE = self._GAME.actionsShape()
        self._LEAF = self._GAME.score(state) is not None
        self._VALID_MOVES_MASK = self._GAME.validMoves(state)

        self._noise = noise
        self._priorComputed = False
        self.W = None

        self._children = np.ndarray(self._ACTIONS_SHAPE, object)
        # (N,W,Q,Prior)
        self._structs = np.zeros((4,) + self._ACTIONS_SHAPE, np.float32)

    def addNetResults(self, rawPrior, W):
        mask = self._VALID_MOVES_MASK
        maskedPrior = rawPrior * mask
        # Normalization by dividing over total sum
        prior = maskedPrior / np.sum(maskedPrior)
        self._priorComputed = True
        self._structs[3] = prior
        self.W = W

    def setNoise(self, noise):
        self._noise = noise

    def _getEffectivePrior(self):
        if not self._priorComputed:
            return
        if self._noise:
            dirichletList = np.ones([np.sum(self._VALID_MOVES_MASK)]) * self._DIRICHLET_PARAMETER
            rawNoise = np.random.dirichlet(dirichletList)
            noise = np.zeros(self._ACTIONS_SHAPE)
            for (i, a) in enumerate(
                    map(lambda x: x[0], filter(lambda x: x[1] == 1, np.ndenumerate(self._VALID_MOVES_MASK)))):
                noise[a] = rawNoise[i]
            return (1 - self._EPSILON) * self.getPriors() + self._EPSILON * noise
        else:
            return self.getPriors()

    def select(self):
        if self._LEAF:
            return None
        Q = self.getQ()
        N = self.getN()
        # Under the assumption that a new noise is used every time.
        P = self._getEffectivePrior()
        U = self._C_PUCT * math.sqrt(0.001 + np.sum(N)) * P / (1 + N)  # Added 0.001 to break ties when N=0
        A = U + Q + 10000 * self._VALID_MOVES_MASK  # To make sure no illegal move is selected
        return np.unravel_index(np.argmax(A), self._ACTIONS_SHAPE)

    def isLeaf(self):
        return self._LEAF

    def getChild(self, a):
        return self._children[a]

    def setChild(self, a, n):
        self._children[a] = n

    def getStruct(self, a):
        rst = np.ndarray([4], np.float32)
        for i in range(4):
            rst[i] = self._structs[i][a]
        return rst

    def setStruct(self, a, s):
        for i in range(4):
            self._structs[i][a] = s[i]

    def getN(self):
        return self._structs[0]

    def getQ(self):
        return self._structs[2]

    def getPriors(self):
        return self._structs[3]

    def getState(self):
        return self._STATE

    def getPlayer(self):
        return self._P

    def isPriorComputed(self):
        return self._priorComputed


class _Worker(object):
    def __init__(self, tree):
        self._TREE = tree
        self._MAX_DEPTH = tree.MAX_SIMULATION_DEPTH
        self._GAME = tree.GAME
        self.root = tree.root
        self._node = None
        self._chain = []

    def _processChain(self):
        chain = self._chain
        W = chain[-1][0].W
        if chain[-1][0].getPlayer() == 1:
            W = -W
        # I will assume the backed up value is the true game outcome when available
        S = self._GAME.score(chain[-1][0].getState())
        if S is not None:
            W = S
        for (node, a) in chain[:-1]:
            s = node.getStruct(a)
            s[0] += 1
            s[1] += (1 if node.getPlayer() == 0 else -1) * W
            s[2] = 1.0 * s[1] / s[0]
            node.setStruct(a, s)
        self._node = None
        self._chain = []

    def work(self, netResult=None):
        # For simple notation
        node = self._node
        chain = self._chain

        # New simulation
        if len(chain) == 0:
            self._node = node = self.root
            self._chain = chain = [(self._node, None)]

        # Prior computed for current node
        if netResult is not None and not node.isPriorComputed():
            node.addNetResults(netResult[0], netResult[1])
        while True:
            # Compute Priors for this state
            if not node.isPriorComputed():
                return node.getState()
            # Simulation done
            if node.isLeaf() or len(chain) == self._MAX_DEPTH:
                self._processChain()
                return None
            # Select step
            a = node.select()
            chain[-1] = (chain[-1][0], a)
            newNode = node.getChild(a)
            if newNode is None:
                newNode = _Node(self._GAME.nextState(node.getState(), a), self._TREE)
                node.setChild(a, newNode)
            self._node = node = newNode
            chain += [(node, None)]


class MCST(object):
    def __init__(self, game, network_inference_function, c_puct, dirichlet_parameter, epsilon, mini_batch_size,
                 maximum_simulation_depth, initial_state=None):
        self.GAME = game
        self._NET_INFERENCE_FUNCTION = network_inference_function
        self.C_PUCT = c_puct
        self.EPSILON = epsilon
        self.DIRICHLET_PARAMETER = dirichlet_parameter
        self.MINI_BATCH_SIZE = mini_batch_size
        self.MAX_SIMULATION_DEPTH = maximum_simulation_depth
        if initial_state is None:
            self.root = _Node(game.initialState(), self, True)
        else:
            self.root = _Node(initial_state, self, True)

    def getTotalN(self):
        return np.sum(self.root.getN())

    def getN(self):
        return self.root.getN()

    def getQ(self):
        return self.root.getQ()

    def getState(self):
        return self.root.getState()

    def getPlayer(self):
        return self.root.getPlayer()

    def simulate(self, number_of_rollouts):
        MB = self.MINI_BATCH_SIZE
        workers = [_Worker(self) for _ in range(MB)]
        networkInput = np.ndarray((MB,) + self.GAME.stateShape())
        networkOutput = [None for _ in range(MB)]
        total_simulated = 0
        workerIndex = 0
        while total_simulated < number_of_rollouts:
            workerMessage = workers[workerIndex].work(networkOutput[workerIndex])
            networkOutput[workerIndex] = None
            if workerMessage is None:
                total_simulated += 1
                continue
            networkInput[workerIndex] = workerMessage
            workerIndex += 1
            if workerIndex == MB:
                networkOutput = self._NET_INFERENCE_FUNCTION(networkInput)
                workerIndex = 0

    def makeMove(self, a):
        c = self.root.getChild(a)
        if c is None:
            s = self.root.getState()
            nextState = self.GAME.nextState(s, a)
            self.root = _Node(nextState, self, True)
        else:
            self.root = c
            c.setNoise(True)
