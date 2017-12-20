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
        self._LEAF = self._GAME.score(state) != None
        self._VALID_MOVES_MASK = self._GAME.validMoves(state)

        self._noise = noise
        self._priorComputed = False
        self.W = None

        self._children = np.ndarray(self._ACTIONS_SHAPE, object)
        # (N,W,Q,Prior,Modified Prior)
        self._structs = np.zeros((5,) + self._ACTIONS_SHAPE, np.float32)

    def addNetResults(self, rawPrior, W):
        mask = self._VALID_MOVES_MASK
        maskedPrior = rawPrior * mask
        # Normalization by dividing over total sum
        prior = maskedPrior / np.sum(maskedPrior)
        self._priorComputed = True
        self._structs[3] = prior
        self.W = W
        self.addNoise(self._noise)

    def addNoise(self, noise):
        self._noise = noise
        if not self._priorComputed:
            return
        if noise:
            numberOfActions = np.prod(self._ACTIONS_SHAPE)
            dirichletList = np.ones([numberOfActions]) * self._DIRICHLET_PARAMETER
            mask = self._VALID_MOVES_MASK
            rawNoise = np.random.dirichlet(dirichletList).reshape(self._ACTIONS_SHAPE)
            maskedNoise = rawNoise * mask
            noise = maskedNoise / np.sum(maskedNoise)
            self._structs[4] = (1 - self._EPSILON) * self._structs[3] + self._EPSILON * noise
        else:
            self._structs[4] = self._structs[3]

    def _sanityCheck(self,X):
        if np.sum(X*(1-self._VALID_MOVES_MASK))>0.001:
            print "HUGE PROBLEM:"
            print self._GAME.stringify(self._STATE)
            print "---"
            print X
            print "_______________________"

    def select(self):
        if self._LEAF:
            return None
        Q = self.getQ()
        N = self.getN()
        P = self.getUsedPriors()
        U = self._C_PUCT * math.sqrt(0.001 + np.sum(N)) * P / (1 + N)  # Added 0.001 to break ties when N=0
        A = U + Q - 100 * (1-self._VALID_MOVES_MASK) #To make sure no illegal move is selected
        return np.unravel_index(np.argmax(A), self._ACTIONS_SHAPE)

    def isLeaf(self):
        return self._LEAF

    def getChild(self, a):
        return self._children[a]

    def setChild(self, a, n):
        self._children[a] = n

    def getStruct(self, a):
        rst = np.ndarray([5], np.float32)
        for i in range(5):
            rst[i] = self._structs[i][a]
        return rst

    def setStruct(self, a, s):
        for i in range(5):
            self._structs[i][a] = s[i]

    def getN(self):
        return self._structs[0]

    def getQ(self):
        return self._structs[2]

    def getTruePriors(self):
        return self._structs[3]

    def getUsedPriors(self):
        return self._structs[4]

    def getState(self):
        return self._STATE

    def getPlayer(self):
        return self._P

    def getPriorComputed(self):
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
        for (node, a) in chain[:-1]:
            s = node.getStruct(a)
            s[0] += 1
            s[1] += W
            s[2] = s[1] / s[0]
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
        if netResult and not node._priorComputed:
            node.addNetResults(netResult[0], netResult[1])
        while True:
            # Compute Priors for this state
            if not node.getPriorComputed():
                return node.getState()
            # Simulation done
            if node.isLeaf() or len(chain) == self._MAX_DEPTH:
                self._processChain()
                return None
            # Select step
            a = node.select()
            chain[-1] = (chain[-1][0], a)
            newNode = node.getChild(a)
            if not newNode:
                newNode = _Node(self._GAME.nextState(node._STATE, a), self._TREE)
                node.setChild(a, newNode)
            self._node = node = newNode
            chain += [(node, None)]


class MCST(object):
    def __init__(self, game, network_inference_function, c_puct, dirichlet_parameter=0.03, epsilon=0.25,
                 mini_batch_size=8, maximum_simulation_depth=500):
        self.GAME = game
        self._NET_INFERENCE_FUNCTION = network_inference_function
        self.C_PUCT = c_puct
        self.EPSILON = epsilon
        self.DIRICHLET_PARAMETER = dirichlet_parameter
        self.MINI_BATCH_SIZE = mini_batch_size
        self.MAX_SIMULATION_DEPTH = maximum_simulation_depth
        self.root = _Node(game.initialState(), self, True)

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
            if type(workerMessage) == type(None):
                total_simulated += 1
                continue
            networkInput[workerIndex] = workerMessage
            workerIndex += 1
            if workerIndex == MB:
                networkOutput = self._NET_INFERENCE_FUNCTION(networkInput)
                workerIndex = 0

    def makeMove(self, a):
        c = self.root.getChild(a)
        if not c:
            self.root = _Node(self.GAME.nextState(self.root, a), self, True)
        else:
            self.root = c
            c.addNoise(True)
