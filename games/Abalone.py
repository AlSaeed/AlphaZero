# coding=utf-8
import numpy as np
import itertools


class Game(object):

    def __init__(self, number_of_history_states=1, no_capture_then_draw_length=100):
        self._NUMBER_OF_HISTORY_STATES = number_of_history_states
        self._NO_CAPTURE_THEN_DRAW_LENGTH = no_capture_then_draw_length
        self._VALID_POSITIONS = {(0, 4), (0, 6), (0, 8), (0, 10), (0, 12), (1, 3), (1, 5), (1, 7), (1, 9), (1, 11),
                                 (1, 13), (2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12), (2, 14), (3, 1), (3, 3),
                                 (3, 5), (3, 7), (3, 9), (3, 11), (3, 13), (3, 15), (4, 0), (4, 2), (4, 4), (4, 6),
                                 (4, 8), (4, 10), (4, 12), (4, 14), (4, 16), (5, 1), (5, 3), (5, 5), (5, 7), (5, 9),
                                 (5, 11), (5, 13), (5, 15), (6, 2), (6, 4), (6, 6), (6, 8), (6, 10), (6, 12), (6, 14),
                                 (7, 3), (7, 5), (7, 7), (7, 9), (7, 11), (7, 13), (8, 4), (8, 6), (8, 8), (8, 10),
                                 (8, 12)}
        self._MOVE_POSITIONS = [[(0, 0)], [(0, 0), (0, 2)], [(0, 0), (0, 2), (0, 4)], [(0, 0), (-1, 1)],
                                [(0, 0), (-1, 1), (-2, 2)], [(0, 0), (-1, -1)], [(0, 0), (-1, -1), (-2, -2)]]
        self._DIRECTION = [(0, -2), (-1, -1), (-1, 1), (0, 2), (1, 1), (1, -1)]

    def _isValidPosition(self, position):
        return position in self._VALID_POSITIONS

    def stateShape(self):
        # history * (white stones | black stones) | current player | moves since last capture
        return 9, 17, self._NUMBER_OF_HISTORY_STATES * 2 + 2

    def actionsShape(self):
        # 7 possible shapes headed in each position:
        #     0           1           2           3           4           5           6
        # . . . . . | . . . . . | . . . . . | . . . . . | . . . . o | . . . . . | o . . . . |
        # . . . . . | . . . . . | . . . . . | . . . o . | . . . o . | . o . . . | . o . . . |
        # . . o . . | . . o o . | . . o o o | . . o . . | . . o . . | . . o . . | . . o . . |
        # . . . . . | . . . . . | . . . . . | . . . . . | . . . . . | . . . . . | . . . . . |
        # . . . . . | . . . . . | . . . . . | . . . . . | . . . . . | . . . . . | . . . . . |
        # 6 possible directions for each move:
        #   0       1       2       3       4       5
        # . . . | . . . | . . . | . . . | . . . | . . . |
        # . ← . | . ↖ . | . ↗ . | . → . | . ↘ . | . ↙ . |
        # . . . | . . . | . . . | . . . | . . . | . . . |
        return 9, 17, 7, 6

    def initialState(self):
        state = np.zeros(self.stateShape(), np.int)
        state[:, :, -2] = 1  # Black is the first to play
        # Positions of balls
        for x in \
                {
                    (0, 4, -4), (0, 6, -4), (0, 8, -4), (0, 10, -4), (0, 12, -4),
                    (1, 3, -4), (1, 5, -4), (1, 7, -4), (1, 9, -4), (1, 11, -4), (1, 13, -4),
                    (2, 6, -4), (2, 8, -4), (2, 10, -4),

                    (8, 4, -3), (8, 6, -3), (8, 8, -3), (8, 10, -3), (8, 12, -3),
                    (7, 3, -3), (7, 5, -3), (7, 7, -3), (7, 9, -3), (7, 11, -3), (7, 13, -3),
                    (6, 6, -3), (6, 8, -3), (6, 10, -3)
                }:
            state[x] = 1
        return state

    def validMoves(self, state):
        isV = self._isValidPosition
        rst = np.zeros(self.actionsShape(), np.int)
        p = self.player(state)
        o = 1 - p
        P = np.copy(state[:, :, -4 + p])  # Player's plane
        O = np.copy(state[:, :, -4 + o])  # Opponent's plane

        # Position contains player's ball
        def _isP(position):
            return isV(position) and P[position] == 1

        # Position contains opponent's ball
        def _isO(position):
            return isV(position) and O[position] == 1

        # Position is valid but empty
        def _isE(position):
            return isV(position) and P[position] == 0 and O[position] == 0

        # Position is not valid
        def _isN(position):
            return not isV(position)

        def _canMoveTwoBalls(forward):
            if forward:
                pos1 = (row + 2 * dy, column + 2 * dx)
                pos2 = (row + 3 * dy, column + 3 * dx)
            else:
                pos1 = (row + 1 * dy, column + 1 * dx)
                pos2 = (row + 2 * dy, column + 2 * dx)
            return _isE(pos1) or (_isO(pos1) and (_isE(pos2) or _isN(pos2)))

        def _canMoveThreeBalls(forward):
            if forward:
                pos1 = (row + 3 * dy, column + 3 * dx)
                pos2 = (row + 4 * dy, column + 4 * dx)
                pos3 = (row + 5 * dy, column + 5 * dx)
            else:
                pos1 = (row + 1 * dy, column + 1 * dx)
                pos2 = (row + 2 * dy, column + 2 * dx)
                pos3 = (row + 3 * dy, column + 3 * dx)
            return _isE(pos1) or (
                    _isO(pos1) and (_isE(pos2) or _isN(pos2) or (_isO(pos2) and (_isE(pos3) or _isN(pos3)))))

        for (position, move, direction) in itertools.product(self._VALID_POSITIONS, range(7), range(6)):
            row, column = position
            dy, dx = self._DIRECTION[direction]
            movePositions = self._MOVE_POSITIONS[move]
            action = (row, column, move, direction)

            # The move positions must contain player's balls
            if not reduce(lambda x, y: x and y,
                          [_isP(q) for q in map(lambda a: (a[0] + position[0], a[1] + position[1]), movePositions)]):
                continue
            # Move forward two balls
            if (move == 1 and direction == 3) or (move == 3 and direction == 2) or (move == 5 and direction == 1):
                rst[action] = _canMoveTwoBalls(True)
            # Move backward two balls
            elif (move == 1 and direction == 0) or (move == 3 and direction == 5) or (move == 5 and direction == 4):
                rst[action] = _canMoveTwoBalls(False)
            # Move forward three balls
            elif (move == 2 and direction == 3) or (move == 4 and direction == 2) or (move == 6 and direction == 1):
                rst[action] = _canMoveThreeBalls(True)
            # Move backward three balls
            elif (move == 2 and direction == 0) or (move == 4 and direction == 5) or (move == 6 and direction == 4):
                rst[action] = _canMoveThreeBalls(False)
            # Side move
            else:
                rst[action] = reduce(lambda x, y: x and y, [_isE(q) for q in map(
                    lambda a: (a[0] + position[0] + dy, a[1] + position[1] + dx), movePositions)])
        return rst

    def nextState(self, state, action):
        isV = self._isValidPosition
        rst = np.ndarray(self.stateShape(), np.int)
        rst[:, :, 0:2 * (self._NUMBER_OF_HISTORY_STATES - 1)] = state[:, :, 2:2 * self._NUMBER_OF_HISTORY_STATES]
        p = self.player(state)
        o = 1 - p
        P = np.copy(state[:, :, -4 + p])  # Player's plane
        O = np.copy(state[:, :, -4 + o])  # Opponent's plane
        opponentsBalls = np.sum(O)  # Number of opponent's balls
        rst[:, :, -2] = o  # Next move is for opponent
        rst[:, :, -1] = state[0, 0, -1] + 1  # Update number of moves since last capture
        row, column, move, direction = action
        dy, dx = self._DIRECTION[direction]
        # Remove balls from their current positions
        for position in self._MOVE_POSITIONS[move]:
            P[row + position[0], column + position[1]] = 0
        # Add balls into their new position
        for position in self._MOVE_POSITIONS[move]:
            P[row + position[0] + dy, column + position[1] + dx] = 1

        def _moveTwoBalls(forward):
            if forward:
                pos1 = (row + 2 * dy, column + 2 * dx)
                pos2 = (row + 3 * dy, column + 3 * dx)
            else:
                pos1 = (row + 1 * dy, column + 1 * dx)
                pos2 = (row + 2 * dy, column + 2 * dx)
            if isV(pos1) and O[pos1] == 1:
                O[pos1] = 0
                if isV(pos2):
                    O[pos2] = 1

        def _moveThreeBalls(forward):
            if forward:
                pos1 = (row + 3 * dy, column + 3 * dx)
                pos2 = (row + 4 * dy, column + 4 * dx)
                pos3 = (row + 5 * dy, column + 5 * dx)
            else:
                pos1 = (row + 1 * dy, column + 1 * dx)
                pos2 = (row + 2 * dy, column + 2 * dx)
                pos3 = (row + 3 * dy, column + 3 * dx)
            if isV(pos1) and O[pos1] == 1:
                O[pos1] = 0
                if isV(pos2):
                    if O[pos2] == 1:
                        if isV(pos3):
                            O[pos3] = 1
                    else:
                        O[pos2] = 1

        # Move forward two balls
        if (move == 1 and direction == 3) or (move == 3 and direction == 2) or (move == 5 and direction == 1):
            _moveTwoBalls(True)
        # Move backward two balls
        elif (move == 1 and direction == 0) or (move == 3 and direction == 5) or (move == 5 and direction == 4):
            _moveTwoBalls(False)
        # Move forward three balls
        elif (move == 2 and direction == 3) or (move == 4 and direction == 2) or (move == 6 and direction == 1):
            _moveThreeBalls(True)
        # Move backward three balls
        elif (move == 2 and direction == 0) or (move == 4 and direction == 5) or (move == 6 and direction == 4):
            _moveThreeBalls(False)
        rst[:, :, -4 + p] = P
        rst[:, :, -4 + o] = O
        # If there was a capture this move then 0 the moves since last capture counter
        if np.sum(O) != opponentsBalls:
            rst[:, :, -1] = 0
        return rst

    def score(self, state):
        if np.sum(state[:, :, -4]) <= 8:
            return -1
        if np.sum(state[:, :, -3]) <= 8:
            return 1
        if state[0, 0, -1] >= self._NO_CAPTURE_THEN_DRAW_LENGTH:
            return 0
        return None

    def player(self, state):
        return state[0, 0, -2]

    def stringify(self, state):
        def _helper(row, x):
            a = 2 * x - 1
            b = (17 - a) / 2
            s = "    "[:b]
            r = ''
            for i in range(a):
                m = b + i
                if i % 2 == 0:
                    r += 'O' if row[m][0] == 1 else '@' if row[m][1] == 1 else '+'
                else:
                    r += ' '
            return s + r

        if np.sum(state[:, :, -4]) <= 8:
            rst = 'Black Won!'
        elif np.sum(state[:, :, -3]) <= 8:
            rst = 'White Won!'
        elif state[0, 0, -1] >= self._NO_CAPTURE_THEN_DRAW_LENGTH:
            rst = 'Draw!'
        else:
            rst = ('Black' if self.player(state) == 1 else 'White') + "'s Turn"
            rst += '\n(' + str(state[0, 0, -1]) + " moves since last capture)"
        rst += '\n'
        for i in range(5):
            rst += _helper(state[i, :, -4:-2], 5 + i) + '\n'
        for i in range(5, 9):
            rst += _helper(state[i, :, -4:-2], 13 - i) + '\n'
        return rst

    def __str__(self):
        return "Abalone (History States: " + str(self._NUMBER_OF_HISTORY_STATES) + \
               ", No Capture then Draw Length: " + str(self._NO_CAPTURE_THEN_DRAW_LENGTH) + ")"
