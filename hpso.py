import random
import time
import numpy as np
import matplotlib.pyplot as plt
R = 4 

"""Some assumptions : 
    we are transmitting same no of bits/sec


"""

total_sn = 400
total_ch = 40
total_ppltn = 50
fitnessValue = []
sensor_nodes = [-1 for i in range(total_sn)] #-1 so that we can compare to see if cluster head has been assigned to this sensor node or not.
enrgyOfSnodes = [10*total_sn] #10 joules of energy for each sensor node
transmitter_energy = 0.2
reciever_energy = 0.2
efs = 0.1 #amplififcation energy for free space model
emp = 0.2 #amplification energy for multi path model
do = 0.123 #threshold transmisson distance (hardcoded) actual=sqrt(efs/emp)
rank=[[0 * total_ch]*total_sn] #a 0 intialized matrix for rank
lambdaa = 0.98 #from pdf

###############################################################Graph
def sample_circle(center):
    a = random.random() * 2 * np.pi
    r = R * np.sqrt(random.random())
    x = center[0]+ (r * np.cos(a))
    y = center[1] + (r * np.sin(a))
    return x,y


def sample_area(center):
    a = random.random()*4
    r = R * np.sqrt(random.random())
    x = center[0]+ (r * np.cos(a))
    y = center[1] + (r * np.sin(a))
    return x,y


for i in range(20):
    # time.sleep(1)
    ps = np.array([sample_circle((0,0)) for i in range(total_sn)])

    """printing the values of coordinates """
    # print(ps)

    plt.plot(ps[:,0],ps[:,1],'.')
    for index,point in enumerate(ps):
        # print(index)
        if(index%10==0):
            plt.plot(point[0]+0.2,point[1]+0.3,'.',color="#000000")
            print(point,"he")

    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.pause(1)
    # plt.ioff()
    plt.show(block=False)
    # plt.close()
    # plt.pause(0.0001)
    plt.clf()

#######################################################################
"""Code Begins here"""

for i in range(total_ppltn):
    for j in range(total_ch):
        population_mtrx[i][j] = int(random.random()*(total_sn-1)+1)
    """call clustering here to calculate the fitness 
    value of this population"""
    fitness[i]=clustering(sensor_nodes,population_mtrx,i)
    #Clustering(sensor_nodes,population_mtrx[i,:])

gBest_ = population_mtrx[fitness.index(max(fitness))] #contains the population consisting of best members
gWorst_ = population_mtrx[fitness.index(min(fitness))]
gBest = gBest_ #initially gBest and gBest' are same
gWorst = gWorst_
#temp = t2 #setting the temprature
temp = 273 #using Kelvin here
gBestFit_ = fitness[fitness.index(max(fitness))] #contains the fitness vlue

while(true):
    for i in range(total_ppltn):
        crnt_generation = population_mtrx[i,:]
        for indx,i in enumerate(crnt_generation):
            r = random.random()
            r1 = random.random()
            r2 = random.random()
            if r<=DR:
                m = len(crnt_generation)-1
                crnt_generation[j] = min(int(abs(crnt_generation[j]+r1*(gBest[j]-crnt_generation[j])-r2*(gWorst[j]-crnt_generation))),m)
        f = clustering(sensor_nodes,population_mtrx,i)#check this
        if f>fitness[i]:
            fitness[i]=f
            population_mtrx[i]=crnt_generation
            if f>gBestFit:
                gBestFit = f
                gBest = crnt_generation
        else:
            if exp((f-f[i])/T) > random.random():
                f[i]=f
                population_mtrx[i]=crnt_generation
    if gBestFit > gBestFit_:
        DR = min(DR/lambdaa,DR_max)
    else:
        DR = max(lambdaa*DR,DR_min)
    gBestFit = gBestFit_
    gBest = gBest_
    #update the generation worst???
    T = lambdaa1 *T
    #termination criteria needed
cluster_heads = gBest

"""Algorithm2"""
def clustering(sensor_nodes,population_mtrx,i):
    cluster_heads = population_mtrx[i]
    for indxOfsn,sn_s in enumerate(sensor_nodes):
        """if value is -1 then it means this sensor has not been assigned a CH
            so we assign it a cluster head"""
        if(sn_s==-1):
            for ch_s in cluster_heads:
                sensor_nodes[indxOfsn]=ch_s  #select the cluster head
                #calculate rank
                rank[indxOfsn,ch_s] = k*LifetimeOfPairs(sn_s,ch_s)*LifetimeOfOne(ch_s)
        #this rank matrix might have old values, need to make new one here
        max_rank = max(rank[indxOfsn])  #find max(rank)
        #choose this cluster head for sensor node
        selected_ch = rank.index(max(rank[indxOfsn]))
        sensor_nodes[indxOfsn] = selected_ch
    return min([LifetimeOfPairs(sensor_node_indx,cluster_head_indx) for sensor_node_indx,cluster_head_indx in zip(len(sensor_nodes),cluster_heads) ])

def LifetimeOfPairs(sensor_node_indx, cluster_head_indx):
    return(enrgyOfSnodes[sensor_node_indx]/eTotal(l,sensor_node_indx,cluster_head_indx))    

def LifetimeOfOne(cluster_head_indx):
    #the clusterhead Communication is a bit complicated 
    #so we need to define it's energy consumption
    return(enrgyOfSnodes[sensor_node_indx]/eTotal(l))

def eTotal(l,sensor_node_indx,cluster_head_indx):
    #l = no of bits to transfer
    #I have assigned constant values here but we need to change it
    #as the values will be different for different parameters
    etx = 0.2
    erx = 0.2
    return(etx+erx)

