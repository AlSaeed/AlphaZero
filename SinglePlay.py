import numpy as np
from MCST import MCST


class SinglePlay(object):

    def __init__(self, game,network_inference_function, c_puct, rollouts_per_move,
                 dirichlet_parameter, epsilon, mcst_mini_batch_size, maximum_simulation_depth,
                 tao,v_resign):
        self._GAME = game
        self._NETWORK_INFERENCE_FUNCTION = network_inference_function
        self._ROLLOUTS_PER_MOVE = rollouts_per_move
        self._C_PUCT = c_puct
        self._DIRICHLET_PARAMETER = dirichlet_parameter
        self._EPSILON = epsilon
        self._MCST_MINI_BATCH_SIZE = mcst_mini_batch_size
        self._MAXIMUM_SIMULATION_DEPTH = maximum_simulation_depth
        self._TAO = tao
        self._V_RESIGN = v_resign

    def move(self, state):
        tree = MCST(self._GAME, self._NETWORK_INFERENCE_FUNCTION, self._C_PUCT, self._DIRICHLET_PARAMETER, self._EPSILON,
                    self._MCST_MINI_BATCH_SIZE, self._MAXIMUM_SIMULATION_DEPTH, state)

        actionsShape = self._GAME.actionsShape()
        actionsNumber = np.prod(actionsShape)
        tao = self._TAO
        v_resign = self._V_RESIGN
        p = tree.getPlayer()
        if tree.root.isLeaf():
            return None

        tree.simulate(self._ROLLOUTS_PER_MOVE)
        if tao == 0:
            N = tree.getN()
            mx = np.max(N)
            pi = (N == mx).astype(np.float32)
            pi /= np.sum(pi)
        else:
            N_tao = np.power(tree.getN(), 1.0/tao)
            pi = N_tao / np.sum(N_tao)
        if v_resign is not None:
            v = np.max(tree.getQ())
            if v < v_resign:
                return None
        a = np.random.choice(actionsNumber, p=pi.reshape([actionsNumber]))
        a = np.unravel_index(a, actionsShape)
        return a
