import numpy as np

class TicTacToe(object):
    def __init__(self):
        pass
    
    def stateShape(self):
        return (3,3,3)
    
    def actionsShape(self):
        return (3,3)

    def initialState(self):
        return np.zeros((3,3,3),dtype=np.int32)

    def validMoves(self,state):
        return 1-np.max(state[:2],axis=0)

    def nextState(self,state,action):
        p = state[2][0][0]
        rst=np.copy(state)
        rst[p][action[0]][action[1]]=1
        rst[2]=1-p
        return rst

    def score(self,state):
        for p in range(2):
            for a in range(2):
                if np.max(np.sum(state[p],a))==3:
                    return 1-2*p
            if state[p][1][1]==1 and ((state[p][0][0]==1  and state[p][2][2]==1)
             or (state[p][0][2]==1 and state[p][2][0]==1)):
                return 1-2*p
        if np.sum(state[:2])==9:
            return 0
        return None
    
    def player(self,state):
        return state[2][0][0]
