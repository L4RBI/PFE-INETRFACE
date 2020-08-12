from tools import *


from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,)) #creating an object Fitness with deap.
creator.create("Bat", list , fitness = creator.Fitness, velocity = list , frequency = None, rate = None, loudness = None, init_rate = None, best = None) #creating an object Agent with deap.
init_R = 0.5
init_A = 0.95

#the function that generates the initial an Bat randomly.
def generate(bmin, bmax, m, histogram, size, init_R, init_A):
    bat = creator.Bat([[random.uniform(bmin,bmax), random.uniform(bmin,numpy.amax(histogram))] for _ in range(size)])
    bat.init_rate = random.uniform(0,init_R)
    bat.rate = bat.init_rate
    bat.loudness = random.uniform(init_A , 2)
    bat.velocity = [[0, 0] for _ in range(size)]
    bat.fitness = toolbox.evaluate(bat, m, histogram = histogram )
    return bat


#updates the position of the agent using the 2, 3 and 4 equations of the paper.
def updateBat(bat, best, fmin, fmax, G, A, m, histogram, size, dim, alpha = 0.9, gamma = 0.9):
    bat.frequency = fmin + (fmax - fmin) * numpy.random.uniform(0,1)
    dis = numpy.multiply(numpy.subtract(bat, best), bat.frequency)
    bat.velocity = list(numpy.add(bat.velocity, dis)) #eauqtion 3 from the paper.
    solution = creator.Bat(list(numpy.add( bat , bat.velocity))) #equation 4 from the paper
   
    rand = numpy.random.random_sample()
    if rand > bat.rate : #random walk using equation 5 from the paper on the global best solution.
        solution = numpy.add(best , [[ random.uniform(-1,1) *A for _ in range(dim)] for _ in range(size)])
        #solution = numpy.add(best , numpy.multiply( A ,[random.gauss(0,1), random.gauss(0,1)]))
        solution = creator.Bat(list(solution)) 

    """ for _ in range(len(solution)): #making sure the bat doesn't go too far and stays in the objective function domain.
        if solution[_] > bmax:
            solution[_]  = bmax
        if solution[_]  < bmin:
            solution[_] = bmin
    """
    solution.fitness = toolbox.evaluate(solution, m, histogram = histogram) #calculation the fitness of the Bat.

    rand =numpy.random.random_sample()
    if bat.fitness >= solution.fitness and rand < bat.loudness : #asserting the solution only if the solution is better and the bat is too loud.

        bat[:] = solution
        bat.fitness = solution.fitness
        bat.loudness = alpha * bat.loudness
        bat.rate = bat.init_rate * (1 - math.exp(-gamma * (G+1)))

    if best.fitness > solution.fitness: #updating the global best.
  
        best[:] = creator.Bat(solution)
        best.fitness = solution.fitness

def Evaluate(bat, m, histogram):
    M = membership(histogram, centers = bat, m = m)
    return (J(histogram, M, bat, m),)

toolbox = base.Toolbox()
toolbox.register("evaluate", Evaluate)

