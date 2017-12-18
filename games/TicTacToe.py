import numpy as np

class Game(object):
    def __init__(self):
        pass
    
    def stateShape(self):
        return (3,3,3)
    
    def actionsShape(self):
        return (3,3)

    def initialState(self):
        return np.zeros((3,3,3),dtype=np.int32)

    def validMoves(self,state):
        return 1-np.max(state[:,:,:2],axis=2)

    def nextState(self,state,action):
        p = state[0][0][2]
        rst=np.copy(state)
        rst[action[0]][action[1]][p]=1
        rst[:,:,2]=1-p
        return rst

    def score(self,state):
        for p in range(2):
            for a in range(2):
                if np.max(np.sum(state[:,:,p],a))==3:
                    return 1-2*p
            if state[1][1][p]==1 and ((state[0][0][p]==1  and state[2][2][p]==1)
             or (state[0][2][p]==1 and state[2][0][p]==1)):
                return 1-2*p
        if np.sum(state[:,:,:2])==9:
            return 0
        return None
    
    def player(self,state):
        return state[0][0][2]

    def stringify(self,state):
        rst = ''
        s = self.score(state)
        if s:
            if s==0:
                rst+= 'Draw!\n'
            elif s==1:
                rst+= 'X Won!\n'
            else:
                rst+= 'O Won!\n'
        else:
            rst+= "Player: "+('X\n' if self.player(state)==0 else 'O\n')
        for i in range(3):
            A=[ 'X' if s[0]==1 else 'O' if s[1]==1 else ' ' for s in state[i] ]
            rst+= A[0]+'|'+A[1]+'|'+A[2]+'\n'
            if i<2:
                rst+= '-----\n'
        return rst
