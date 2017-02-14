import queue as Q
import sys
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

class State(object):
    
        def __init__(self,pos,cost,parent=None,direction=""):
            self.pos     = pos
            self.h_cost  = 1
            self.parent  = parent
            self.possible     = [-1,1,-3,3]
            self.goalState =[1,2,3,4,5,6,7,8,9]
            self.blank_pos =-1
            if parent != None:    
                    self.path=parent.path+direction
            else:
                    self.path=""
            if parent != None:  
                    self.cost = parent.cost+1
            else:
                    self.cost = 0
            return
        
        def calculatePathCost(self):
                if self.parent != None:
                    self.g_cost=self.parent.g_cost+1
                else:
                    self.g_cost = 0
                return
        def possibleMoves(self):
             self.blank_pos = self.pos.index(9)+1
             if self.blank_pos %n ==1 :
                   self.possible [0] =0
             if self.blank_pos %n ==0 :
                    self.possible [1] =0
             if self.blank_pos /n ==1 :
                    self.possible [2] =0
             if self.blank_pos /n >=n-1:
                    self.possible [3] =0
             return
                    #4 new states
        
        def calculate_h_cost(self):
            h_cost = 0
            for i in range(len(self.pos)):
                h_cost = h_cost+abs(self.pos[i]/3-(i+1)/3)+abs(self.pos[i]%3-(i+1)%3)
                print("index",i)
            return h_cost
        
        def isgoalState(self):
            print ("checking for goal state")
            if self.h_cost == 0 and self.pos == self.goalState:
                    return True
            else :
                    return False
        def __cmp__(self, other):
                return self.cost > other.cost  - self.cost <= other.cost 

        def generateStates (self):
             print ("inside generate states")
             self.possibleMoves()
             temppos=list(self.pos)
             if self.possible[0] != 0:
                 temppos[self.blank_pos],temppos[self.blank_pos+self.possible[0]]=temppos[blank_pos+self.possible[0]],temppos[blank_pos]
                 #(self,pos,cost,parent,blank_pos)
                         #,self.blank_pos+self.possible[0]
                 state=State(temppos,0,self,'L')
                 state.cost=state.cost+state.calculate_h_cost()
                 queue.push(state,state.cost)
             temppos=list(self.pos)
             if self.possible[1] != 0:
                 temppos[self.blank_pos],temppos[self.blank_pos+self.possible[1]]=temppos[self.blank_pos+self.possible[1]],temppos[self.blank_pos]
                 #self.blank_pos+self.possible[1],
                 state = State(temppos,0,self,'R')
                 state.cost=state.cost+state.calculate_h_cost()
                 queue.push(state,state.cost)
             temppos=list(self.pos)
             if self.possible[2] != 0:
                 temppos[self.blank_pos],temppos[self.blank_pos+self.possible[2]]=temppos[self.blank_pos+self.possible[2]],temppos[self.blank_pos]
                 #self.blank_pos+self.possible[2],
                 state=State(temppos,0,self,'U')
                 state.cost=state.cost+state.calculate_h_cost()
                 queue.push(state,state.cost)
             temppos=list(self.pos)
             if self.possible[3] != 0:
                 temppos[self.blank_pos],temppos[self.blank_pos+self.possible[3]]=temppos[self.blank_pos+self.possible[3]],temppos[self.blank_pos]
                 #self.blank_pos+self.possible[3],
                 state=State(temppos,0,self,'D')
                 state.cost=state.cost+state.calculate_h_cost()
                 queue.push(state,state.cost)
             return 
queue = PriorityQueue()
intial =[1,2,3,4,5,9,7,8,6]
blank_pos = intial.index(9)+1
state= State(intial,0)
queue.push(state,0)
n=3
#queue.push((5,Skill(5, 'Proficient')))
while queue:
    current=queue.pop()
    goalCheck=current.isgoalState()
    if (goalCheck==True):
            print ("ss"+current.path)
            break
    current.generateStates()
  #  break
    
