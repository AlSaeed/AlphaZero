import random
import numpy as np
# Helpful to test small games like Tic-Tac-Toe.


class MiniMaxPlayer(object):
    def __init__(self, game):
        self._GAME = game
        self._STATE_VALUE = {}
        self._OPTIMAL_MOVES = {}
        self.ALL_STATES = []
        self._stateValue(game.initialState())

    def _stateValue(self, state):
        key = str(state)
        if key in self._STATE_VALUE:
            return self._STATE_VALUE[key]
        g = self._GAME
        p = g.player(state)
        score = g.score(state)
        if score is not None:
            self._STATE_VALUE[key] = score
            self._OPTIMAL_MOVES[key] = []
            self.ALL_STATES += [state]
            return score
        validMoves = [x[0] for x in filter(lambda x: x[1] == 1, np.ndenumerate(g.validMoves(state)))]
        scores = [self._stateValue(g.nextState(state, move)) for move in validMoves]
        best = max(scores) if p == 0 else min(scores)
        moves = []
        for m in range(len(validMoves)):
            if scores[m] == best:
                moves += [validMoves[m]]
        self._OPTIMAL_MOVES[key] = moves
        self._STATE_VALUE[key] = best
        self.ALL_STATES += [state]
        return best

    def optimalMoves(self, state):
        return self._OPTIMAL_MOVES[str(state)]

    def move(self, state):
        A = self.optimalMoves(state)
        return A[random.randint(0, len(A) - 1)]
