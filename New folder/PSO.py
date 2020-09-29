#import operator
import random

from tools import *


from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness = creator.Fitness, velocity = list, best = None)

#initialize the Particule randomly
def generate( pmin, pmax , vmin, vmax, dim, size, histogram):
    particule = creator.Particle([[random.uniform(pmin,pmax), random.uniform(pmin,numpy.amax(histogram))] for _ in range(size)])
    particule.velocity = [[random.uniform(vmin, vmax) for _ in range(dim)] for _ in range(len(particule))]
    return particule
    
#Update the postion of a particule using the equations 3 and 4 provided by the paper.
def updateParticle(particule, best, constant1, constant2, weight, vmin, vmax, dim, data):
    rand1 = [[random.uniform(0, 1) for _ in range(dim)] for _ in range(len(particule))]
    rand2 = [[random.uniform(0, 1) for _ in range(dim)] for _ in range(len(particule))]
    rand1 = numpy.multiply(rand1, constant1)
    rand2 = numpy.multiply(rand2, constant2)
    v = numpy.multiply(particule.velocity, weight)#(_ * weight for _ in particule.velocity) 
    rand1_local = numpy.multiply(rand1,numpy.subtract(particule.best,particule))#map(operator.mul, rand1, map(operator.sub, particule.best, particule))
    rand2_global = numpy.multiply(rand2,numpy.subtract(best,particule))#map(operator.mul, rand2, map(operator.sub, best, particule))
    particule.velocity =  numpy.add( v, numpy.add( rand1_local, rand2_global))#map(operator.add, v, map(operator.add, rand1_local, rand2_global)) #equation 3
  
    for i in range(len(particule.velocity)):
        for _ , velocity in enumerate(particule.velocity[i]): #making sure the velocity isn't yoo high.
            if abs(velocity) < vmin:
                particule.velocity[i][_] = math.copysign(vmin, velocity)
            elif abs(velocity) > vmax:
                particule.velocity[i][_] = math.copysign(vmax, velocity)
    
    particule[:] = numpy.add(particule, particule.velocity) #equation 4 
    particule.fitness.values = toolbox.evaluate(data = data, particule = particule) #calculating the fitness

    if particule.best.fitness < particule.fitness: #updating the personal best.
                particule.best = creator.Particle(particule)
                particule.best.fitness.values = particule.fitness.values

def Evaluate(data, particule, m):
    M = membership(data, centers = particule, m = m)
    return (J(data, M, particule, m),)

toolbox = base.Toolbox()
toolbox.register("evaluate", Evaluate, m = 2)
