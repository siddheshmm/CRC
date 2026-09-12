import numpy as np
from deap import base, creator, tools, algorithms
import random, time
from scipy.stats import sem, t
from scripts.runLongMemoryTask import longMemoryTask
from scripts.runShortMemoryTask import shortMemoryTask

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
class geneticAlgo:
    def __init__(self, dg, seed, num_nodes, edges, logger, moleculeInflux, memoryTask, tau=0, runTime=0, repetition=0):
        """
        Setting instance variables of the genetic algorithm for the network parameter optimization.

        Args:
            dg (tuple): (molecules, rules) used for the construction of derivation graph in MOD.
            seed (int): seed value for the random and np.random libraries
            num_nodes (int): number of nodes.
            edges (int): number of edges.
            logger (logging.Logger): logger file for intermediate steps.
            moleculeInflux (int): the fixed molecule amount to be used in inflow signal generation.
            memoryTask (string): choosing the memory tasks for regression step.
            tau (int): past input time, tau, should be defined if long memory task is chosen.
            runTime (int): simulation run time
            repetition (int): repetition value in input signal for the input hold time.
        """

        self.logger=logger
        self.tau=tau
        self.memoryTask=memoryTask
        self.runTime=runTime
        self.repetition=repetition
        self.moleculeInflux=moleculeInflux
        self.DG = dg
        self.SEED = seed
        self.moleculeAmount = num_nodes
        self.edges = edges
        self.INT_BOUNDS = {
            'seedValue': (1, 1),
            'scalingRateValues': (1,10),
            'totalMoleculeAmount': (self.moleculeAmount, self.moleculeAmount),
            'reactionScale': (1,10),
        }
        self.FLOAT_BOUNDS =  {'outFlowValue': (0.05, 1.0)}
        self.reactionRatesBounds = (10, 100)
        self.numEdges = lambda: int(self.edges)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialIndividuals)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.objective)
    
    def run_simulation(self, seed, outFlow, scalingRate, moleculeAmount, reactionScale, reactionRates):
        """
        Calling stochastic simulation for the topology and input parameters.

        Args:
            seed (int): seed value for the random and np.random libraries
            outFlow (float): the amount to exist system over the time.
            scalingRate (int): scaling rate for the input signal.
            moleculeAmount (int): molecule amount to be used in input signal generation.
            reactionScale (int): scaling rate for the reaction rates.
            reactionRates (list[float]): the edge weights for each reaction.
        Returns:
            float: the lowest normalized root mean square error from network optimization process.
        """

        seed = int(seed)
        outFlow = float(outFlow)
        scalingRate = int(scalingRate)
        moleculeAmount = int(moleculeAmount)
        reactionScale = int(reactionScale)
        reactionRates = [float(weight) for weight in reactionRates]
        reactionRates_str = ','.join(map(str, reactionRates))
        arguments = [seed, outFlow, scalingRate, moleculeAmount, reactionScale, reactionRates_str]
        if self.memoryTask=="short":
            topologyEvaluator = shortMemoryTask(self.SEED, self.DG, self.moleculeInflux, arguments, runTime=self.runTime, repetition=self.repetition)
        else:
            topologyEvaluator = longMemoryTask(self.SEED, self.DG, self.tau, self.moleculeInflux, arguments, runTime=self.runTime, repetition=self.repetition)

        return topologyEvaluator.main()
    
    def compute_confidence_interval(self, scores, confidence=0.95):
        """
        Computing the confidence interval for the scores obtained from multi-calls of simulation function.

        Args:
            scores (list): list of minimum root mean square errors obtained from the simulation calls.
            confidence (float): confidence value for the standard error calculation.
        Returns:
            tuple: mean and margin values for the input scores.
        """

        n = len(scores)
        mean_score = np.mean(scores)
        std_error = sem(scores)
        h = std_error * t.ppf((1 + confidence) / 2, n - 1)
        return mean_score, h

    def objective(self, params, numRuns=2, confidence=0.95):
        """
        Objective function for the genetic algorithm. Running the stochastic simulation and regression to obtain
        lowest root mean square error.

        Args:
            params (list): input parameters for the simulation.
            numRuns (int): number of runs for the simulation.
            confidence (float): confidence value for the standard error calculation.
        Returns:
            float: error value obtained by confidence interval calculation.
        """
        seed, outFlow, scalingRate, moleculeAmount, reactionScale, *reactionRates = params
        reactionRates = [float(round(float(weight / 100), 3)) for weight in reactionRates]
        scores = [self.run_simulation(seed, outFlow, scalingRate, moleculeAmount, reactionScale, reactionRates) for _ in range(numRuns)]
        mean_score, margin_of_error = self.compute_confidence_interval(scores, confidence)
        return mean_score + margin_of_error,
    
    def initialIndividuals(self):
        """
        Initializing the parameters to be used in the objective function.

        Returns:
            list: input parameters for the objective function.
        """
        seed = random.randint(*self.INT_BOUNDS['seedValue'])
        outFlow = round(random.uniform(*self.FLOAT_BOUNDS['outFlowValue']), 3)
        scalingRate = random.randint(*self.INT_BOUNDS['scalingRateValues'])
        moleculeAmount = self.moleculeAmount
        reactionScale = random.randint(*self.INT_BOUNDS['reactionScale'])
        num_edges = self.numEdges()
        reactionRates = [random.randint(*self.reactionRatesBounds) for _ in range(num_edges)]
        return [seed, outFlow, scalingRate, moleculeAmount, reactionScale] + reactionRates
    
    def mutateIntegers(self, individual, indpb):
        """
        Mutation function for integer values.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        integer_indices = [0, 2, 3, 4]
        for i in integer_indices:
            if random.random() < indpb:
                step = 1
                key = list(self.INT_BOUNDS.keys())[integer_indices.index(i)]
                low, high = self.INT_BOUNDS[key]
                individual[i] = max(low, min(high, individual[i] + random.choice([-step, step])))
        return individual
    
    def mutateFloatValues(self, individual, indpb):
        """
        Mutation function for float values.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        if random.random() < indpb:
            step = 0.2
            low, high = self.FLOAT_BOUNDS['outFlowValue']
            individual[1] = max(low, min(high, individual[1] + random.choice([-step, step])))
        individual[1] = round(individual[1], 3)
        return individual
    
    def mutateReactionRates(self, individual, indpb):
        """
        Mutation function for reaction rates.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        low, high = self.reactionRatesBounds
        numEdges = self.numEdges()
        for i in range(5, 5 + numEdges):
            if random.random() < indpb:
                step = 10
                individual[i] = max(low, min(high, individual[i] + random.choice([-step, step])))
        return individual
    
    def custom_mutate(self, individual, indpb=0.5):
        """
        Main custom mutation function, calling sub mutation functions for the integer, float and edge-weights values.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        individual = self.mutateIntegers(individual, indpb)
        individual = self.mutateFloatValues(individual, indpb)
        individual = self.mutateReactionRates(individual, indpb)
        return individual,
    
    def main(self, populationSize=4, eliteSize=2, maxGenerations=10, maxTime=500):
        """
        Main function of the network optimization genetic algorithm.

        Args:
            populationSize (int): genetic algorithm population size
            eliteSize (int): genetic algorithm elite size
            maxGenerations (int): number of generations
            maxTime (int): maximum run time of the genetic algorithm.

        Returns:
            float: the lowest normalized root mean square error from network optimization process.
        """
        population = self.toolbox.population(n=populationSize)
        startTime = time.time()
        prevMinValue=0.0
        currentMinValue=0.0
        for gen in range(maxGenerations):
            elapsedTime = time.time() - startTime
            if elapsedTime >= maxTime:
                self.logger.info("Network Optimization - Elapsed time reached, stopping.")
                break
            elites = tools.selBest(population, eliteSize)
            population = algorithms.eaSimple(population, self.toolbox, cxpb=1.0, mutpb=0.5, ngen=1, verbose=False)[0]
            population.extend(elites)
            population = tools.selBest(population, len(population) - eliteSize)
            fits = [ind.fitness.values[0] for ind in population]
            self.logger.info(f"Network Optimization - Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            currentMinValue=min(fits)
            if gen==0:
                prevMinValue=currentMinValue
            else:
                self.logger.info(f"Network Optimization - currentMinValue: {currentMinValue}")
            prevMinValue=currentMinValue
        bestIndividual = tools.selBest(population, 1)[0]
        weights = bestIndividual[5:]
        reactionRates = [float(round(float(weight / 100), 3)) for weight in weights]
        bestIndividual[5:]=reactionRates
        self.logger.info(f"Network Optimization - Best parameters: {bestIndividual}")
        return currentMinValue
    
