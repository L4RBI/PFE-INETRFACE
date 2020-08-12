from tools import *

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,)) #creating an object Fitness with deap.
creator.create("Agent", list, fitness = creator.Fitness, velocity = list) #creating an object Agent with deap.

def S(r, f, l):
    temp = numpy.multiply(r,-1)
    return (f * numpy.exp(temp / l)) - numpy.exp(temp)

#
def compute_c(g, max_iter):
    cmax = 1
    cmin = 0.00000001
    return cmax - g * ((cmax - cmin) / max_iter)

#calculates the "{ }" part of the 2.7 equation of the paper.
def compute_braquet(agent1 , agent2, c, f, l, u, lb, ubound, lbound, size):#the braquet of the 2.7 equation
    distance = []
    for i in range(len(agent1)):
        distance.append(Dis(agent1[i],agent2[i]))
    #right = (agent1 - agent2) / distance 
    right = numpy.subtract(agent2, agent1)
    rd = numpy.zeros(right.shape)
    for i in range(len(rd)):
        rd [i] = right[i] / distance[i]
    function = numpy.add(numpy.mod(distance, 2), 2) #norm the distance
    s_thing = S(function, f = f, l = l)
    t = numpy.zeros((size,2))
    for i in range(size):
        t = numpy.multiply([(((ubound - lbound) * c) / 2), (((u - lb) * c) / 2) ], s_thing[i])
    return list( t * rd)


def generate(histogram, ubound, lbound, size, u, lb, m):
    agent = creator.Agent([random.uniform(ubound, lbound), random.uniform(u, lb)] for _ in range(size))
    agent.best = agent
    agent.fitness = toolbox.evaluate(agent = agent, data = histogram, m = m)
    return agent

def updateGrassHopper(agent , Population, c, f, l, best, histogram, u, lb, ubound, lbound, size, m):
    sigma = numpy.zeros((size,2))
    for p in Population:
        if not numpy.array_equal(p, agent):
            sigma += numpy.multiply(random.random() , compute_braquet(agent1 = agent, agent2 = p, c = c, f = f, l = l, lb = lb, u = u, ubound = ubound, lbound = lbound, size = size)) # the sum of the "{ }" part of equation 2.7.
    b = numpy.multiply(best, [[random.random(), random.random()] for _ in range(size)]) #randomizing the second term
    sigma = sigma * [[random.random(),random.random()]]
    sigma = list(c * sigma  + b) #the result with radomization of both terms.
    
    agent[:] = sigma #updating the postion.
    agent.fitness = toolbox.evaluate(agent = agent, data = histogram, m = m)
  
def Evaluate(data, agent, m):
    M = membership(data, centers = agent, m = m)
    return (J(data, M, agent, m = m),)

#setting up the functions for easier calls using the toolbox provided by deap
toolbox = base.Toolbox()
toolbox.register("evaluate", Evaluate) #the function used to calculate the fitness set and evaluate.

