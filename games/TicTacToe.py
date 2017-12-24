import numpy as np


class Game(object):

    def stateShape(self):
        return (3, 3, 3)

    def actionsShape(self):
        return (3, 3)

    def initialState(self):
        return np.zeros((3, 3, 3), dtype=np.int32)

    def validMoves(self, state):
        return 1 - np.max(state[:, :, :2], axis=2)

    def nextState(self, state, action):
        player = state[0][0][2]
        result = np.copy(state)
        result[action[0]][action[1]][player] = 1
        result[:, :, 2] = 1 - player
        return result

    def score(self, state):
        for player in range(2):
            for axis in range(2):
                if np.max(np.sum(state[:, :, player], axis)) == 3:
                    return 1 - 2 * player
            if state[1][1][player] == 1 and ((state[0][0][player] == 1 and state[2][2][player] == 1)
                                             or (state[0][2][player] == 1 and state[2][0][player] == 1)):
                return 1 - 2 * player
        if np.sum(state[:, :, :2]) == 9:
            return 0
        return None

    def player(self, state):
        return state[0][0][2]

    def stringify(self, state):
        result = ''
        score = self.score(state)
        if score is not None:
            if score == 0:
                result += 'Draw!\n'
            elif score == 1:
                result += 'X Won!\n'
            else:
                result += 'O Won!\n'
        else:
            result += "Player: " + ('X\n' if self.player(state) == 0 else 'O\n')
        for row in range(3):
            cells = ['X' if score[0] == 1 else 'O' if score[1] == 1 else ' ' for score in state[row]]
            result += cells[0] + '|' + cells[1] + '|' + cells[2] + '\n'
            if row < 2:
                result += '-----\n'
        return result

    def __str__(self):
        return "Tic Tac Toe"
