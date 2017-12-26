# Networks:

## Properties:

Network Implementation should:

* Follow inference/loss/training design pattern.
* Accept both inputs & outputs of shapes identical to the game it is used for.

## Implementation:

Each Game class must implement the following functions:

* **`inference(self, states)`**: Takes as input placeholder of shape `(None,)+game.stateShape` and return as output a tuple `(PH, VH)` where `PH` is a tensor of the logits of policy with shape `(None,)+game.actionsShape` while `VH` is a tensor of values with shape `(None,1)`.
* **`loss(self, p_logits, w, pi, z)`**: Takes as input 4 tensors corresponding to logits of network's policy, values returned by network, mcst search probabilities, & actual outcomes of games respectively. It returns the loss operation.
* **`training(self, loss, learning_rate)`**: Takes as input the loss & the learning rate and returns a training operation.
* **`__str__(self)`**: the returned string must include all parameters used in defining the network (e.g. number of residual blocks, regularization factor, etc.).