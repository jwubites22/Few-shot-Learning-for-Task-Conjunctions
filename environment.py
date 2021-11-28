from math import exp
import gym
from gym.core import RewardWrapper
from numpy.core.fromnumeric import argmax
from gym_gridworlds.envs.gridworld_env import  *
import random
from random import randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(threshold=np.inf)
#add terminal states in goal orientated
#min of 2 q matrices 
#min of 2 q matrices and use value 3 when they disagreee
def checkDone(env,goals,currentState,times):
    
    if currentState in goals:
        if times==1:
            env.done=True
            return times
        else:
            times=+1
            return times
    else:
        return 0
def checkError(q,q_prev,number,row,width):
    if number==0:
        
        for i in range (row):
            for j in range(width):
                v1=q[i][j]
                v2=q_prev[i][j]
                if ((abs(v1-v2)>1e-5)):
                    return False
        return True
    else:
        for x in range(number):
            for i in range (row):
                for j in range(5):
                        v1=q[x][i][j]
                        v2=q_prev[x][i][j]
                        if abs(v1-v2)>1e-5:
                            return False
        return True

def checkDoneQ(env,goals,currentState,times):
    
    for i in range(len(goals)):
        
        if currentState ==goals[i]:
 
            if times[i]==1:
                
                env.done=True
                return times
            else:
                times=np.zeros(len(goals))
                times[i]=1
                return times
        
    return np.zeros(len(goals))

def checkQRow(Q,currentState,n):
    maxI=0
    maxValue=-10
    for i in range(5):
        value=Q[currentState][i]
        if maxValue<value:
            maxI=i
            maxValue=value
    return maxI

def checkCount(count,max):
    if count<max:
        return True
    return False

def minQ(q1,q2,number,n):
    q_matrix=np.zeros(((n)*(n),5))
    if number==0:
        for i in range(n*n):
            for j in range(5):
                if q1[i][j]>q2[i][j]:
                    q_matrix[i][j]=q2[i][j]
                else:
                    q_matrix[i][j]=q1[i][j]
        return q_matrix
    else:
        q_matrix=np.zeros(((n)*(n),5))
        here=np.argmax(q1,axis=1)
        here1=np.argmax(q2,axis=1)
        
        for i in range(n*n):
            if here[i]!=here1[i]:
                for j in range(5):
                    if q1[i][j]>q2[i][j]:
                         q_matrix[i][j]=q2[i][j]+3
                    else:
                        q_matrix[i][j]=q1[i][j]+3
                
            else:
                for j in range(5):
                    if q1[i][j]>q2[i][j]:
                         q_matrix[i][j]=q2[i][j]
                    else:
                        q_matrix[i][j]=q1[i][j]

        return q_matrix



def q_learning(env,start,goal,n,number,q_matrix1):
    learning_rate=0.1
    gamma=1#discount rate
    epsilon=0.1  #exploration rate         # Exponential decay rate for exploration prob
    if(number==0):
        q_matrix=np.zeros(((n)*(n),env.action_space.n))
        for i in range(n*n):
            if i not in goal:
                q_matrix[i][4]=-100
    else:
        q_matrix=np.zeros(((n)*(n),env.action_space.n))
        for i in range(n*n):
                for j in range(5):
                    q_matrix[i][j]=q_matrix1[i][j]
        
    
    current_state=start
    count=0
    times=0
    action=0
    q_prev=np.ones(((n)*(n),env.action_space.n))*100
    converged=False
    reward=0
    rewardTracker=[]
    counter=[]
    max_episode=10000
    prevRandom=0
  
    randomStart=np.zeros(((n)*(n)))
    #converge when difference between q matrix is below certain threshold
    while converged!=True:
      
        if(times!=1):
            times=0
        #up right down left stay
        exp_exp_tradeoff=random.uniform(0,1) 
        if (times==1):
            
            if exp_exp_tradeoff>epsilon:
                action=checkQRow(q_matrix,current_state,n)
               
           
            else:
                action=randrange(5)
        else:
            if exp_exp_tradeoff>epsilon:
            
                action=checkQRow(q_matrix,current_state,n)
               
            else:
                action=randrange(4)
        next_state = np.where(env.P[action][current_state]==1)[0][0]
        #print("1",q_matrix[current_state][action])
        q_prev1=q_matrix[current_state][action]
        if action==4:
            reward=reward+2
            q_matrix[current_state][action] = q_matrix[current_state][action]+learning_rate*(2-q_matrix[current_state][action])
        else:
            reward=reward-0.1
            q_matrix[current_state][action] = q_matrix[current_state][action]+learning_rate*(-0.1+gamma*(np.max(q_matrix[next_state]))-q_matrix[current_state][action])
        current_state=next_state
        times=checkDone(env,goal,current_state,times)
        if env.done==True and times==1 and action==4:
            times=0
            rewardTracker.append(reward)
            counter.append(count)
            count+=1
            reward=0
            #action=4 
            current_state=randrange(n*n)
            
            while(prevRandom==current_state):
                current_state=randrange(n*n)
            prevRandom=current_state
            if current_state in goal:
                times=1
            randomStart[current_state]=randomStart[current_state]+1
            env.done=False
            converged=checkError(q_prev,q_matrix,0,n*n,5)
            for i in range (n*n):
                for j in range(5):
                        q_prev[i][j]=q_matrix[i][j]
    return q_matrix,count,counter,rewardTracker
def checkQMatrix(Q,goal,currentState):
    maxI=0
    maxValue=-10
    value=0
    maxValueAction=-1
    for i in range(len(goal)):

        value=np.max(Q[i][currentState])
        
        if maxValue<value:
            maxI=i
            maxValue=value
    maxValueAction=np.argmax(Q[maxI][currentState])
    if(value==0):
        return randrange(5)
    else:
        return maxValueAction

def goalOrientated_q_Learning(env,start,goal,n,terminal_state):
    learning_rate=0.1
    gamma=1 #discount rate
    epsilon=1 #exploration rate
           # Exponential decay rate for exploration prob
    q_matrix=np.zeros((len(terminal_state),n*n,env.action_space.n))
    q_prev=q_matrix+1
    visitedGoals=[]
    current_state=start
    max_episode=1000
    count=0
    times=0
    action=0
    converged=False
    reward=0
    rewardTracker=[]
    counter=[]
    prev_state=0
    while converged!=True:
        
        
        exp_exp_tradeoff=random.uniform(0,1)
        #up right down left stay
        
        if (times==1):
        
            if exp_exp_tradeoff>epsilon:
                    
                    action=checkQMatrix(q_matrix,goal,current_state)
            
            else:
                action=randrange(5)
        else:
            if exp_exp_tradeoff>epsilon:
                    action=checkQMatrix(q_matrix,goal,current_state)
            else:
                action=randrange(4)
        next_state = np.where(env.P[action][current_state]==1)[0][0]
        for i in range(len(visitedGoals)):
            #fix
            if next_state in terminal_state and action==4:
                if next_state != visitedGoals[i]:   
                    delta=-10
                    reward=reward-10
                else:
                    if next_state == visitedGoals[i] and next_state in goal:
                        delta=2-q_matrix[i][current_state][action]
                        reward=reward+2
                    else:
                        delta=-0.1-q_matrix[i][current_state][action]
                        reward=reward-0.1
            else:
                    reward=reward-1
                    delta=-0.1+gamma*np.max(q_matrix[i][next_state])-q_matrix[i][current_state][action]
                
            q_matrix[i][current_state][action] = q_matrix[i][current_state][action]+learning_rate*delta  
        current_state=next_state
        times=checkDone(env,terminal_state,current_state,times)
        if times==1 and current_state not in visitedGoals and action==4:
            visitedGoals.append(current_state)
        if env.done==True and times==1 and action==4:
            times=0
            rewardTracker.append(reward)
            counter.append(count)
            count+=1
            reward=0
            #action=4
            
            prev_state=current_state
            current_state=randrange(n*n)
            while(current_state==prev_state):
                current_state=randrange(n*n)
            prev_state=current_state
            if current_state in terminal_state:
                times=1
            env.done=False
            converged=checkError(q_prev,q_matrix,1,n*n,5)
            for x in range(len(terminal_state)):
                for i in range (n*n):
                    for j in range(5):
                        q_prev[x][i][j]=q_matrix[x][i][j]

    return q_matrix,count,counter,rewardTracker

def printQValue(q_matrix,n):
    
    valueRow=np.zeros((n,n))
    here=np.max(q_matrix,axis=1)
    for i in range(n):
        for j in range(n):
            valueRow[i][j]=here[(i*n)+j]
    ax = sns.heatmap(valueRow, linewidth=1)
    plt.show()


#up right down left stay
#0 1 2 3 
#4 5 6 7
#8 9 10 11
#12 13 14 10
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd



walls=[]
start=0
average=1
colGoal=[]
rowGoal=[]
qMinGoal=[]
conGoal=[]
goalOrientatedGoal=[]
size=[]
steps=1
count1x=np.zeros((average,steps))
countx=np.zeros((average,steps))
count2x=np.zeros((average,steps))
count3x=np.zeros((average,steps))
count4x=np.zeros((average,steps))
for y in range(steps): 
    goals=[]
    n=4*(y+1)  #grid soze
    terminal_state=[]
    rowGoals=[]
    colGoals=[]
    finalGoal=[(n*n)-1]
    for i in range(n):
        colGoals.append((n-1)+((i)*n))
        terminal_state.append((n-1)+((i)*n))
    for i in range(n):
        if ((n*n)-1)-i in colGoals:
            rowGoals.append(((n*n)-1)-i)
        else:
            rowGoals.append(((n*n)-1)-i)
            terminal_state.append(((n*n)-1)-i)
   
    env = GridworldEnv(goals=goals,start=0,walls=walls,n=n+2)

    for x in range(average): 
        
        # substitute environment's name


        q_matrixColumn,count,counter,reward=q_learning(env,0,rowGoals,n,0,0)
        countx[x][y]=count
    
        
        q_matrixRow,count1,counter1,reward1=q_learning(env,0,colGoals,n,0,0)
        count1x[x][y]=count1

        qMin=minQ(q_matrixColumn,q_matrixRow,0,n)

        qMinFinal,count2,counter2,reward2=q_learning(env,0,finalGoal,n,1,qMin)
        count2x[x][y]=count2

        q_matrixCon,count3,counter3,reward3=q_learning(env,0,[(n*n)-1],n,0,0)
        count3x[x][y]=count3

        q_matrixGoal,count4,counter4,reward4=goalOrientated_q_Learning(env,0,colGoals,n,terminal_state)
        count4x[x][y]=count4

   
    # rowGoal.append(countx/average)
    # colGoal.append(count1x/average)
    # qMinGoal.append(count2x/average)
    # conGoal.append(count3x/average)
    # goalOrientatedGoal.append(count4x/average)
    size.append(n)
    countxVar=countx.std(0)
    count1xVar=count1x.std(0)
    count2xVar=count2x.std(0)
    count3xVar=count3x.std(0)
    count4xVar=count4x.std(0)
    

    file_path = 'randomfile.txt'
    sys.stdout = open(file_path, "w")
    print("rowgoals average Episode",countx.mean(0))
    print("colgoals average Episode",count1x.mean(0))
    print("qMin average Episode",count2x.mean(0))
    print("conjunction average Episode",count3x.mean(0))
    print("goalOrientated average Episode",count4x.mean(0))
    print("size \n",size)
    print("rowgoals std \n",countxVar)
    print("colgoals std \n",count1xVar)
    print("qMin average std \n",count2xVar)
    print("conjunction std \n",count3xVar)
    print("goalOrientated std \n",count4xVar)



printQValue(q_matrixColumn,n)
print("row goals episode",count)
plt.plot(counter,reward)

plt.xlabel('sample',fontsize=10)
plt.ylabel('reward',fontsize=10)

plt.show()

printQValue(q_matrixRow,n)
print("col goals episode",count1)
plt.plot(counter1,reward1)
plt.xlabel('sample',fontsize=10)
plt.ylabel('reward',fontsize=10)

plt.show()

printQValue(qMin,n)


printQValue(qMinFinal,n)
print("qMin episode",count2)
plt.plot(counter2,reward2)
plt.xlabel('sample',fontsize=10)
plt.ylabel('reward',fontsize=10)

plt.show()

printQValue(q_matrixCon,n)
print("conjunction episodes",count3)
plt.plot(counter3,reward3)
plt.title('Conjunction')
plt.xlabel('counter')
plt.ylabel('reward')
plt.show()

valueRow=np.zeros((n,n))
valueCol=np.zeros((n,n))
for x in range(len(terminal_state)):
    here=np.max(q_matrixGoal[x],axis=1)
    for i in range(n):
        for j in range(n):
            
                valueCol[i][j]=here[(i*n)+j]
    
    ax = sns.heatmap(valueCol, linewidth=1)
    plt.show()     

plt.plot(counter4,reward4)
plt.title('goalOrientated')
plt.xlabel('episode')
plt.ylabel('reward ')
plt.show()
