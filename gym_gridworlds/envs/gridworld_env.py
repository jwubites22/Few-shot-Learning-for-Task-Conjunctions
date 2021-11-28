import gym
from gym import spaces
from gym.core import ObservationWrapper
import numpy
import math



class GridworldEnv(gym.Env):
    reward_range = (-1, 2)
    action_space = spaces.Discrete(5)
    # although there are 2 terminal squares in the grid
    # they are considered as 1 state
    # therefore observation is between 0 and 14
    
    def __init__(self,goals=None,start=None,walls=None,n=None):
        self.observation_space = spaces.Discrete(n*n)
        self.start_pos=start
        self.done=False
        self.gridworld = numpy.arange(
                self.observation_space.n
                ).reshape(n,n)
        counter=0
        for i in range(n):
            for j in range(n):
                if(i==0 or j==0 or j==n-1 or i==n-1):
                    self.gridworld[i][j]=-1
                else:
                     self.gridworld[i][j]=counter
                     counter=counter+1

        # state transition matrix
        self.P = numpy.zeros((self.action_space.n,
                              (n-2)*(n-2),
                              (n-2)*(n-2)))
        # any action taken in terminal state has no effect
        

        for s in self.gridworld.flat[0:  self.observation_space.n]:
           
                row, col = numpy.argwhere(self.gridworld == s)[0]
                if(self.gridworld[row][col]!=-1):
                    for a, d in zip(
                            range(self.action_space.n),
                            [(-1, 0), (0, 1), (1, 0), (0, -1),(0,0)]
                            ):
                        next_row = max(0, min(row + d[0], n-2))
                        next_col = max(0, min(col + d[1], n-2))
                        if self.gridworld[next_row][next_col]==-1:
                            if next_row==0 or next_row==n-1:
                                next_row=row
                            elif next_col==0 or next_col==n-1:
                                next_col=col
                        s_prime = self.gridworld[next_row, next_col]
                        self.P[a, s, s_prime] = 1

        self.goals=goals
        self.R = numpy.full((self.action_space.n,
                             (n-2)*(n-2)), -0.1)

def addPunnishmentState(self,wall,n):
    for i in range(len(wall)) :
        if(wall[i]+n<=n*n-1):
            self.R[0,wall[i]+n]=-1
        if(wall[i]-n>=0):
            self.R[2,wall[i]-n]=-1
        if(wall[i]-1>=0):
            self.R[1,wall[i]-1]=-1
        if(wall[i]+1<=n*n-1):
            self.R[3,wall[i]+1]=-1
    return self

def step(self, action):
        assert self.action_space.contains(action)
        
        action = self.pertube_action(action)
        
        g = [None,None] # #virtual state 
        if self.state in self.T_states:
            return str(g), self._get_reward(self.state, action), True, None
        elif action == 4 and ([self.position,self.position] in self.T_states) and not self.state[0]:
            new_state = [self.position, self.position]
        else:
            x, y = self.state[1]
            if action == 0:
                x = x - 1
            elif action == 1:
                x = x + 1
            elif action == 2:
                y = y + 1
            elif action == 3:
                y = y - 1
            self.position = (x, y)
            new_state = [None, self.position]