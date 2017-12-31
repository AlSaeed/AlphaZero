import sys
from Trainer import Trainer

if __name__ == '__main__':
    execfile('configs/'+sys.argv[1])
    trainer = Trainer(
                        EXPERIMENT_NAME,
                        GAME,
                        NETWORK,
                        C_PUCT,
                        NUMBER_OF_MINIBATCHES,
                        LEARNING_RATE_POLICY_STRING,
                        MINIBATCH_SIZE,
                        GAMES_BETWEEN_VERSIONS,
                        MINIBATCHES_BETWEEN_VERSIONS,
                        SNAPSHOT_PERIOD,
                        SELECTION_LOOKBACK_SPAN,
                        INDEPENDENT_SIMULATION_POLICY,
                        ROLLOUTS_PER_MOVE,
                        DIRICHLET_PARAMETER,
                        EPSILON,
                        MCST_MINIBATCH_SIZE,
                        MAXIMUM_SIMULATION_DEPTH,
                        TAO_FUNCTION_STRING
                    )
    trainer.train()