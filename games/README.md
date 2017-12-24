# Games:

## Properties:

All Games must be:

* Two players
* Zero sum
* Finite
* Deterministic
* Has unique starting state

## Implementation:

Each Game class must implement the following functions:

* **`stateShape(self)`**: returns a tuple of the shape of state representation.
* **`actionsShape(self)`**: returns a tuple of the shape of actions representation.
* **`initialState(self)`**: returns a numpy ndarray - of shape `stateShape(self)` - representing the initial state of the game.
* **`validMoves(self, state)`**: returns a numpy ndarray - of shape `actionsShape(self)` - having 1 in the position of each valid move. It can assume `state` is a valid non-leaf game state.
* **`nextState(self, state, action)`**: returns a numpy ndarray - of shape `stateShape(self)` - representing the state resulting from make move indexed `action` in `state`. It can assume `state` is a valid non-leaf game state & `action` is a valid move in `state`.
* **`score(self, state)`**: returns a scalar score in the range [-1, 1] from the point view of type 0 if `state` is leaf game state & None otherwise. It can assume `state` is a valid game state.
* **`player(self, state)`**: returns an integer in {0, 1} representing the index of the player taking turn at `state`. It can assume `state` is a valid non-leaf game state.
* **`stringify(self, state)`**: returns a human-friendly string representing `state`. It can assume `state` is a valid game state.
* **`__str__(self)`**: the returned string must include all parameters used in defining the object (e.g. size of the board, number of history steps included in the state representation, etc.).