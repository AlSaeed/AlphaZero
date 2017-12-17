import numpy as np
import math

class SearchNode(object):
    def __init__(self,TREE,STATE,noise=False):
        self.GAME=TREE.GAME
        self.STATE=STATE
        self.ACTIONS_SHAPE=self.GAME.actionsShape()
        self.CHILDREN=np.ndarray(self.ACTIONS_SHAPE,object)
        #(N,W,Q,Prior,Modified Prior)
        self.STRUCTS=np.zeros((5,)+self.ACTIONS_SHAPE,np.float32)
        self.DIR_PARAM=TREE.DIR_PARAM
        self.EPSILON=TREE.EPSILON
        self.C_PUCT=TREE.C_PUCT
        self.LEAF=self.GAME.score(STATE)!=None
        self.noise=noise
        self.P_MULT= 1 if self.GAME.player(STATE)==0 else -1
        self.PRIORS_COMPUTED=False
        self.W=None
        
    def addPrior(self,networkOutput,W):
        networkOutput*=self.GAME.validMoves(self.STATE)
        Priors=networkOutput/np.sum(networkOutput)
        self.PRIORS_COMPUTED=True
        self.STRUCTS[3]=Priors
        self.W=W
        self.addNoise(self.noise)

    def addNoise(self,noise):
        self.noise=noise
        if noise:
            D = np.ones([np.prod(self.ACTIONS_SHAPE)])*self.DIR_PARAM
            D = np.random.dirichlet(D).reshape(self.ACTIONS_SHAPE)
            self.STRUCTS[4]=(1-self.EPSILON)*self.STRUCTS[3]+self.EPSILON*D
        else:
            self.STRUCTS[4]=self.STRUCTS[3]

    def select(self):
        if self.LEAF:
            return None
        Q = self.STRUCTS[2]
        N = self.STRUCTS[0]
        P = self.STRUCTS[4]
        U = self.C_PUCT*math.sqrt(0.001+np.sum(N))*P/(1+N)#Added 0.001 to break ties when N=0
        A = U + Q
        return np.unravel_index(np.argmax(A),self.ACTIONS_SHAPE)

    def getChild(self,a):
        return self.CHILDREN[a]

    def setChild(self,a,n):
        self.CHILDREN[a]=n

    def getStruct(self,a):
        rst = np.zeros([5])
        for i in range(5):
            rst[i]=self.STRUCTS[i][a]
        return rst

    def setStruct(self,a,s):
        for i in range(5):
            self.STRUCTS[i][a]=s[i]

class SearchWorker(object):
    def __init__(self,TREE):
        self.TREE=TREE
        self.MAX_DEPTH=TREE.MAX_SIM_DEPTH
        self.GAME=TREE.GAME
        self.root=TREE.root
        self.node=None
        self.chain=[]

    def processChain(self):
        chain=self.chain
        W = chain[-1][0].W
        for (node,a) in chain[:-1]:
            s = node.getStruct(a)
            s[0]+=1
            s[1]+=W
            s[2]=s[1]/s[0]
            node.setStruct(a,s)
        self.node=None
        self.chain=[]
        
    def work(self, RST=None):
        #For simple notation
        node = self.node
        chain = self.chain
        #New simulation
        if not node:
            self.node=node=self.root
            self.chain=chain=[(self.node,None)]

        #Priors computed for current node
        if RST and not node.PRIORS_COMPUTED:
            node.addPrior(RST[0],RST[1])
        while True:
            #Compute Priors for this state
            if not node.PRIORS_COMPUTED:
                return node.STATE
            #Simulation done
            if node.LEAF or len(chain)==self.MAX_DEPTH:
                self.processChain()
                return None
            #Select step
            a = node.select()
            chain[-1]=(chain[-1][0],a)
            newNode = node.getChild(a)
            if not newNode:
                newNode=SearchNode(self.TREE,self.GAME.nextState(node.STATE,a))
                node.setChild(a,newNode)
            self.node=node=newNode
            chain+=[(node,None)]

    

class MCST(object):
    def __init__(self,GAME,NET_INF,C_PUCT,DIR_PARAM=0.03,EPSILON=0.25,MINI_BATCH=8,MAX_SIM_DEPTH=500):
        self.GAME=GAME
        self.NET_INF=NET_INF
        self.C_PUCT=C_PUCT
        self.EPSILON=EPSILON
        self.DIR_PARAM=DIR_PARAM
        self.MINI_BATCH=MINI_BATCH
        self.MAX_SIM_DEPTH=MAX_SIM_DEPTH
        self.root= SearchNode(self,GAME.initialState(),True)
            
    def getTotalN(self):
        return np.sum(self.root.STRUCTS[0])

    def getN(self):
        return self.root.STRUCTS[0]

    def simulate(self,N):
        MB = self.MINI_BATCH
        workers = [ SearchWorker(self) for _ in range(MB) ]
        netInp = np.ndarray((MB,)+self.GAME.stateShape())
        netOut = [None for _ in range(MB)]
        n=0
        i=0
        while n<N:
            r = workers[i].work(netOut[i])
            netOut[i]=None
            if type(r)==type(None):
                n+=1
                continue
            netInp[i]=r
            i+=1
            if i==MB:
                netOut=self.NET_INF(netInp)
                i=0

    def makeMove(self,a):
        c = self.root.getChild(a)
        if not c:
            self.root= SearchNode(self,GAME.nextState(self.root,a),True)
        else:
            self.root = c
            c.addNoise(True)

