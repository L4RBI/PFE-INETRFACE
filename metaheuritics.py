import BAT
import PSO
import GAO
import datetime
import numpy

from deap import tools
from deap import base


class metaheuristics:
    def __init__(self, path, size, m, dim=2, M=255):
        self.path = path
        self.size = size
        self.dim = dim
        self.m = m
        self.M = M
        self.histogram = BAT.Histogram(path)

    def bat(self, N, GEN, bmin=0, bmax=255, fmin=0, fmax=0.2):
        toolbox = base.Toolbox()
        toolbox.register("bat", BAT.generate, m=self.m, bmin=bmin, bmax=bmax,
                         histogram=self.histogram, size=self.size, init_R=0.5, init_A=0.95)
        toolbox.register("population", tools.initRepeat, list, toolbox.bat)
        toolbox.register("update", BAT.updateBat, size=self.size, dim=self.dim)
        Population = toolbox.population(n=N)
        mean_A = 0
        A = 0
        best = None
        for bat in Population:  # finding the initial global best
            if not best or best.fitness > bat.fitness:
                best = BAT.creator.Bat(bat)
                best.fitness = bat.fitness
            A += bat.loudness
        mean_A = A / N
        begin_time = datetime.datetime.now()
        for G in range(GEN):  # computing the BA GEN times
            A = 0
            # updating the Bat postion one by one and opdating the Global best as well.
            for bat in Population:
                toolbox.update(bat, best=best, G=G, A=mean_A, m=self.m,
                               fmin=fmin, fmax=fmax, histogram=self.histogram)
                A += bat.loudness
            mean_A = A / N
        x = datetime.datetime.now() - begin_time
        return best, x

    def pso(self, N, GEN, pmin=0, pmax=255, vmin=-100, vmax=100, constant1=2, constant2=2, weight=0.5):
        toolbox = base.Toolbox()
        toolbox.register("particle", PSO.generate, pmin=pmin, pmax=pmax, vmin=vmin,
                         vmax=vmax, dim=self.dim, size=self.size, histogram=self.histogram)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.particle)
        toolbox.register("update", PSO.updateParticle, constant1=constant1, constant2=constant2,
                         weight=weight, vmin=vmin, vmax=vmax, dim=self.dim, data=self.histogram)
        Population = toolbox.population(n=N)
        best = None
        for particule in Population:
            particule.fitness.values = PSO.toolbox.evaluate(
                data=self.histogram, particule=particule)
        begin_time = datetime.datetime.now()
        for g in range(GEN):  # computing the PSO Algorithm GEN times
            for particule in Population:
                if not particule.best or particule.best.fitness < particule.fitness:
                    particule.best = PSO.creator.Particle(particule)
                    particule.best.fitness.values = particule.fitness.values
                if not best or best.fitness < particule.fitness:  # updating the global best.
                    best = PSO.creator.Particle(particule)
                    best.fitness.values = particule.fitness.values

            for particule in Population:  # updating the position for each particule.
                toolbox.update(particule, best)
        x = datetime.datetime.now() - begin_time
        return best, x

    def gao(self, N, GEN, ubound=0, lbound=255, lb=0, f=0.5, l=1.5):
        u = numpy.amax(self.histogram)
        toolbox = base.Toolbox()
        # setting agent as the function that initialize the agent with the function generate and default args.
        toolbox.register("agent", GAO.generate, histogram=self.histogram,
                         ubound=ubound, lbound=lbound, u=u, lb=lb, size=self.size, m=self.m)
        # intrepeat helps with repeating the call of the function n times.
        toolbox.register("swarm", tools.initRepeat, list, toolbox.agent)
        # registering the updateGrassHopper function as update with setting the default values for some of the args.
        toolbox.register("update", GAO.updateGrassHopper, f=f, l=l, histogram=self.histogram,
                         size=self.size, m=self.m, ubound=ubound, lbound=lbound, u=u, lb=lb)
        Swarm = toolbox.swarm(n=N)  # intializing the swarm with n agents.
        best = None  # initializing the best as none.
        for agent in Swarm:  # finding the best position for the initial swarm/population
            if not best or best.fitness > agent.fitness:
                best = GAO.creator.Agent(agent)
                best.fitness = agent.fitness
        for g in range(GEN):  # runing the GAO Gen times
            c = GAO.compute_c(g + 1, GEN)  # computing c
            for agent in Swarm:  # updating each agent using the best solution found
                toolbox.update(agent=agent, Population=Swarm, c=c, best=best)
            for agent in Swarm:  # updating the best solution after updating the whole swarm.
                if not best or best.fitness > agent.fitness:
                    best = GAO.creator.Agent(agent)
                    best.fitness = agent.fitness
                    print("new best:", best)
        return best
