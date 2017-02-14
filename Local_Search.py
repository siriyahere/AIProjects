import sys
import heapq
import copy
from queue import *
from operator import itemgetter

class PriorityQueue:
    def __init__(self):
        self._queue = []
        #self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, item))
        #self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]


class Node:

    def __init__(self,id,neighbors=[]):
        self.color=-1
        self.neighbors=[]
        self.vertexid=id
        self.domain= [x for x in range(K)]

def check_if_assigned(vertex):
    for i in range(N):
        if vertex[i].color ==-1:
            #print ("success")
            return i
    return -1

def valid_assignment(vertex,next_vertex,color):
    for ver in next_vertex.neighbors:
        if vertex[ver].color == color :
            return False
    return True

def min_value_heuristic(next_vertex,domain=[]):
    count=0
    for i in range(K):
        if(valid_assignment(next_vertex,i)):
            #domain.append(i)
            count=count+1
    return count


def check_if_vertex_not_assigned(vertex):
    if assignment[vertex]==-1:
        return vertex
    return -1


def get_mrv(vertex,domain):

    #return_vertex =-1
    #count = len(vertex.domain)
    count = len(domain.get(vertex))
    #return K-count
    return count

def get_ordered_value(graph,vertex,domain):


    max = 0
    n=len(vertex.neighbors)
    select_value=[]
    id = vertex.vertexid
    loc=domain.get(id)
    for val in loc:
        clen=0
        count=K
        all_assigned= True
        for ver in vertex.neighbors:
            if assignment[ver]==-1:
                all_assigned = False
                #domain=ver.domain.copy()
                if val in domain[ver] :
                    clen = len(domain[ver])-1
                else:
                    clen = len(domain[ver])
                #count= count+clen
                if clen < count:
                    count=clen
        #if count==K and all_assigned:
         #   count=n*K
        select_value.append((val,count))
    sorted(select_value, key=itemgetter(1),reverse=True)
    return select_value


def get_max_uncolored_neighbors(graph,vertex):
    count =0
    for ver in vertex.neighbors:
        if assignment[ver] == -1:
            count = count+1
    return count


class HInfo:
    def __init__(self,vertex,mrv,max_neigh):
        self.vertex=vertex
        self.mrv_vertex=mrv
        self.min_vertex=max_neigh
        self.domain=[]
    def __lt__(self, other):
        return self.mrv_vertex < other.mrv_vertex - self.mrv_vertex > other.mrv_vertex

    def __eq__(self, other):
        if isinstance(other, HInfo):
            return self.vertex == other.vertex # and (self.bar == other.bar))
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.__repr__())


def remove_inconsistency(vertex1,vertex2,domain):
    removed=False
    #domain=vertex1.domain
    local_domain=domain.get(vertex1)
    for x in local_domain:
            if x in domain.get(vertex2) and len(domain.get(vertex2)) == 1:
                #vertex1.domain.remove(x)
                domain[vertex1].remove(x)
                removed=True
                break
    if len(domain.get(vertex1)) ==0 or len(domain.get(vertex2)) == 0:
        removed = True
    return removed


class Arc:
    def __init__(self, one, two):
        self.one = one
        self.two = two


def remove_arc_inconsistency(vertex,domain):
    lqueue = PriorityQueue()
    i=0
    for ver in csp:
        j,k=ver
        lqueue.push(Arc(j,k),i)
        i=i+1
    while lqueue:
        try:
            adj=lqueue.pop()
            if remove_inconsistency(adj.one,adj.two,domain):
                if (len(domain.get(adj.one)) == 1):
                    for ver in vertex[adj.one].neighbors:
                        lqueue.push(Arc(ver,adj.one),i)
                        i=i+1
        except:
          #  print("exception")
            return True
    return True


def get_next_vertex_for_assigning(vertex,domain):

    selected_vertex =-1
    minimum = K+1
    # for i in range(N):
    #
    #     next_vertex=check_if_vertex_not_assigned(i)
    #     if(next_vertex!=-1):
    #         mrv_vertex_dlength =get_mrv(next_vertex,domain)
    #         if mrv_vertex_dlength < minimum :
    #             selected_vertex=next_vertex
    #             minimum = mrv_vertex_dlength
    for i in range(N):
        if assignment[i] == -1:
            leng = len(domain.get(i))
            if leng <minimum:
                minimum=leng#(domain.get(i))
                selected_vertex=i
            elif leng == minimum:
                if len(vertex[selected_vertex].neighbors) < len(vertex[i].neighbors):
                    minimum = leng
                    selected_vertex = i

    return selected_vertex
    #return -1


def remove_domain(assigned,i,domain):
    domain[assigned]=[i]
   # for color in domain[assigned]:
    #    if i != color:
    #        domain[assigned].remove(color)


def add_domain(assigned,removed):
    for color in removed:
        assigned.domain.append(color)


def add_to_neighbor(graph,vertex,i):
    for n in vertex.neighbors:
        if graph[n].color ==-1 :
            graph[n].domain.append(i)


def remove_from_neighbor(vertex,i,domain):
    for n in vertex.neighbors:
        if assignment[n] ==-1 and i in domain.get(n):
            domain.get(n).remove(i)
            if len(domain.get(n)) == 0:
                return False
    return True


def dfs_bt_opti(N,M,K,vertex,domain):
    next_vertex = get_next_vertex_for_assigning(vertex,domain)
    if next_vertex == -1:
        return True
    values=get_ordered_value(vertex,vertex[next_vertex],domain)
    domain_backup = copy.deepcopy(domain)
    for (i,clen) in values:
        if clen != 0:
            #vertex[next_vertex].color=i
            assignment[next_vertex]=i
            global assess
            assess=assess+1
            #if assess > 135:
            #    print(domain[next_vertex],assess,next_vertex,i,clen)

            remove_domain(next_vertex,i,domain)
            res=remove_from_neighbor(vertex[next_vertex],i,domain)
            if res :
                res= remove_arc_inconsistency(vertex,domain)
            #else:
              #  print("failure due to forward checking",assess,next_vertex)
            #    # copied=True
            if res:
                res = dfs_bt_opti(N,M,K,vertex,domain)
            #else:
                #print("failure due to arc inconsistency issues",assess)
            if res:
               # print("successful assessment", assess,next_vertex,i)
                return res
            else:
              #  print("failed assessment dfs opti ", assess, next_vertex, i, clen)
                domain =copy.deepcopy(domain_backup)
                assignment[next_vertex]=-1
                #vertex[next_vertex].color = -1
                assess = assess - 1
    #domain = copy.deepcopy(domain_backup)
    return False


def dfs_bt(N,M,K,vertex):
    next_vertex=check_if_assigned(vertex)
    if next_vertex==-1:
        return True
    for i in range(K):
        if valid_assignment(vertex[next_vertex],i):
            vertex[next_vertex].color=i
            res=dfs_bt(N,M,K,vertex)
            if res:
                return res
            vertex[next_vertex].color=-1
    return False


if __name__ == '__main__':
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    no_of_rows = -1
    inp = []
    searchResult = ""
    input_file = open("backtrack_hard")
    #output_file = sys.argv[4
    lines = [i.rstrip() for i in input_file.readlines()]
    vars=lines[0].split('\t')
    N = int(vars[0])
    M = int(vars[1])
    K = int(vars[2])
    assignment = [-1 for i in range(N)]
    assess = 0
    vertex = []
    domain = {}
    values =[i for i in range(K)]
    for i in range(N):
        vertex.append(Node(i))
        domain[i]=copy.deepcopy(values)
    cnt=0
    csp=[]

    for line in lines:
        if cnt ==0:
            cnt=cnt+1
            continue
        vars=line.split('\t')
        vertex[int(vars[0])].neighbors.append(int(vars[1]))
        vertex[int(vars[1])].neighbors.append(int(vars[0]))
        csp.append((int(vars[0]),int(vars[1])))
        csp.append((int(vars[1]), int(vars[0])))
    print("calling opti")
    dfs_bt_opti(N,M,K,vertex,domain)
    for i in range(N):
        print (i,assignment[i])
