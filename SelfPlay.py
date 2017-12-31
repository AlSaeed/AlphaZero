import numpy as np
from MCST import MCST


class SelfPlay(object):

    def __init__(self, game, c_puct, independent_simulation_policy, rollouts_per_move,
                 dirichlet_parameter, epsilon, mcst_mini_batch_size, maximum_simulation_depth,
                 tao_function):
        self._GAME = game
        self._ROLLOUTS_PER_MOVE = rollouts_per_move
        self._C_PUCT = c_puct
        self._INDEPENDENT_SIMULATION_POLICY = independent_simulation_policy
        self._DIRICHLET_PARAMETER = dirichlet_parameter
        self._EPSILON = epsilon
        self._MCST_MINI_BATCH_SIZE = mcst_mini_batch_size
        self._MAXIMUM_SIMULATION_DEPTH = maximum_simulation_depth
        self._TAO_FUNCTION = tao_function

    def _playGame(self, network_inference_function, v_resign):
        tree = MCST(self._GAME, network_inference_function, self._C_PUCT, self._DIRICHLET_PARAMETER, self._EPSILON,
                    self._MCST_MINI_BATCH_SIZE, self._MAXIMUM_SIMULATION_DEPTH)

        actionsShape = self._GAME.actionsShape()
        actionsNumber = np.prod(actionsShape)
        min_values = [10, 10]
        memo = []
        moveDepth = 0
        while True:
            tao = self._TAO_FUNCTION(moveDepth)
            moveDepth += 1
            p = tree.getPlayer()
            state = tree.getState()
            if tree.root.isLeaf():
                z = self._GAME.score(state)
                # Game ended due to max depth reached
                if not z:
                    z = 0
                # for now assuming leaf node is also added
                memo += [(state, np.ones(actionsShape, np.float32) / actionsNumber, p)]
                break

            if self._INDEPENDENT_SIMULATION_POLICY:
                tree.simulate(self._ROLLOUTS_PER_MOVE)
            else:
                tree.simulate(self._ROLLOUTS_PER_MOVE - tree.getTotalN())

            N_tao = np.power(tree.getN(), 1.0/tao)
            pi = N_tao / np.sum(N_tao)
            v = np.max(tree.getQ())
            if v_resign is not None:
                if v < v_resign:
                    z = -1 if p == 0 else 1
                    memo += [(state, pi, p)]
                    break
            else:
                min_values[p] = min(v, min_values[p])
            memo += [(state, pi, p)]
            a = np.random.choice(actionsNumber, p=pi.reshape([actionsNumber]))
            a = np.unravel_index(a, actionsShape)
            tree.makeMove(a)
        return map(lambda m: (m[0], m[1], z if m[2] == 0 else -z), memo), \
               None if v_resign is not None else (z, min_values[0], min_values[1])

    def _determineVResign(self, v_list):
        v_lost, v_others, v_all = [], [], []
        for (z, v1, v2) in v_list:
            if z == 0:
                v_others += [v1, v2]
            elif z < 0:
                v_lost += [v1]
                v_others += [v2]
            else:
                v_lost += [v2]
                v_others += [v1]
            v_all += [v1, v2]

        v_lost.sort()
        v_others.sort()
        v_all.sort()
        v_lost += [None]
        v_others += [None]
        lost_head, others_head, n_lost, n_others = 0, 0, 0, 0
        v_resign = None
        for i in range(len(v_all)):
            if v_all[i] == v_lost[lost_head]:
                n_lost += 1
                lost_head += 1
            else:
                n_others += 1
                others_head += 1
            if 1.0 * n_others / (n_others + n_lost) <= 0.05:
                v_resign = v_all[i]
        return v_resign if v_resign is not None else -10

    def play(self, network_inference_function, number_of_games):
        data = []
        v_games = number_of_games / 10
        v_list = []
        # for _ in range(v_games):
        #     d, v = self._playGame(network_inference_function, None)
        #     data += [d]
        #     v_list += [v]
        # v_resign = self._determineVResign(v_list)
        # for _ in range(number_of_games - v_games):
        #     d, v = self._playGame(network_inference_function, v_resign)
        #     data += [d]

        #For now no resignation
        for _ in range(number_of_games):
            d, v = self._playGame(network_inference_function, None)
            data += [d]
            v_list += [v]
        return data
