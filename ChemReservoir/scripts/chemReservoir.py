import mod
from mod import *
from deap import base, creator, tools, algorithms
import random, time
from scripts.networkParameterOptimization import geneticAlgo

def generateRingWithFixedDistancedChords(nodes, chordLength, chordStep):
    """
    Generating cycle topologies with chords.

    Args:
        nodes (int): number of nodes
        chordLength (int): chord length
        chordStep (int): chord step

    Returns:
        list: Edge list of a reservoir topology
    """
    edges = [(i, (i + 1)) for i in range(nodes)]
    edges.append((nodes, 0))
    edges.extend(
        [(i, ((i + chordLength) % (nodes + 1))) if (i + chordLength) > nodes else (i, (i + chordLength)) for i in
         range(0, nodes + 1, chordStep)])
    return edges

def generateLabels(nodes):
    """
    Generating node labels.

    Args:
        nodes (int): number of nodes

    Returns:
        list: Node labels.
    """
    labels = []
    for i in range(1, nodes + 1):
        labels.append(str(i))
    return labels

def generateMolecules(nodes):
    """
    Generating node labels.

    Args:
        nodes (int): number of nodes

    Returns:
        list: Node labels.
    """
    graphs = []
    labels = generateLabels(nodes)
    for i in range(nodes):
        graphs.append(Graph.fromDFS("[" + labels[i] + "]", name=labels[i]))
    return graphs

def defineModRules(nodes, edges):
    """
    Defining list of MOD rules for the input edge list.
    These rules are used for the construction of MOD chemical reaction network.

    Args:
        nodes (int): number of nodes
        edges (list): edge list

    Returns:
        list: MOD rule list.
    """
    rules = []
    labels = generateLabels(nodes)
    rules.append(Rule.fromDFS("[" + str(0) + "]>>[" + labels[0] + "]"))
    for i in edges:
        rules.append(Rule.fromDFS("[" + str(i[0]) + "]>>[" + str(i[1]) + "]"))
    return rules

def generateInputGraphs(nodes):
    """
    Generating input graphs (molecules) for MOD StochSim.
    Each node represents a pseudo-molecule, graph, in the network.

    Args:
        nodes (int): number of nodes

    Returns:
        list: MOD input graph list.
    """
    molecules = [Graph.fromDFS("[" + str(0) + "]", name=str(0))]
    molecules.extend(generateMolecules(nodes))
    return molecules

def buildNetwork(nodes, length, step):
    """
    Building the MOD network for MOD-StochSim.
    Networks are represented by tuples (molecules, rules),
    list of input graphs and reaction rules.

    Args:
        nodes (int): number of nodes
        length (int): chord length
        step (int): chord step

    Returns:
        tuple: MOD input graphs and reaction rules.
    """
    edges = generateRingWithFixedDistancedChords(nodes, length, step)
    rules = defineModRules(nodes, edges)
    molecules = generateInputGraphs(nodes)
    return molecules, rules


class chemReservoir:
    def __init__(self, inputSeed, inputLogger, memoryTask=None, tau=None, runTime=None, repetition=None, networkOptMaxTime=None):
        """
        Setting instance variables of the genetic algorithm for topology selection optimization.

        Args:
            inputSeed (int): seed value for the random and np.random libraries
            inputLogger (logging.Logger): logger file for intermediate steps.
            memoryTask (string): choosing the memory tasks for regression step.
            tau (int): past input time, tau, should be defined if long memory task is chosen.
        """

        self.logger = inputLogger #logger file for storing genetic algorithm steps.
        self.seed = inputSeed
        self.memoryTaskValues=(memoryTask, tau)
        self.runTime=runTime
        self.repetition=repetition
        self.networkOptimizationMaxTime = networkOptMaxTime
        random.seed(inputSeed) #seed for random library for reproducibility.
        self.INT_BOUNDS=None
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initializeIndividuals)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.customMutate, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.objective)

    def callGeneticAlgorithm(self, dg, nodes, numEdges, moleculeAmount, memoryTask, runTime=None, repetition=None, networkOptTime=None, tau=0):
        """
        Calling the genetic algorithm for the optimal topology selection.

        Args:
            dg (tuple): (molecules, rules) used for the construction of derivation graph in MOD.
            nodes (int): number of nodes.
            numEdges (int): number of edges.
            tau (int): time lag value.
            moleculeAmount (int): the fixed molecule amount to be used in inflow signal generation.
            memoryTask (string): choosing the memory tasks for regression step.
            runTime (int): simulation run time
            repetition (int): repetition value in input signal for the input hold time.
            networkOptTime (int): max run time for network optimization process.
            tau (int): past input time, tau, should be defined if long memory task is chosen.

        Returns:
            float: the lowest normalized root mean square error from topology selection.
        """
        genetic = geneticAlgo(dg, self.seed, nodes + 1, numEdges, self.logger,
                              moleculeAmount, memoryTask, tau=tau, runTime=runTime, repetition=repetition) #+1 due to the food molecule, indexed 0.

        error = genetic.main(maxTime=networkOptTime)
        return error

    def objective(self, params):
        """
        Objective function of the genetic algorithm for topology selection.

        Args:
            params (list):

        Returns:
            float: the lowest normalized root mean square error from topology selection.
        """
        nodes, moleculeInflow, distance, step = params
        network = buildNetwork(nodes, distance, step)
        score = self.callGeneticAlgorithm(network, nodes, len(network[1]), moleculeInflow, self.memoryTaskValues[0], runTime=self.runTime, repetition=self.repetition, networkOptTime=self.networkOptimizationMaxTime, tau=self.memoryTaskValues[1])
        return score,  # , due to the deap library.

    def initializeIndividuals(self):
        """
        Initializing the random initial individuals for genetic algorithm.

        Returns:
            list: List of random initial individuals.
        """
        initials = []
        for i, (key, (low, high)) in enumerate(self.INT_BOUNDS.items()):
            step = 10 if i < 2 else 2
            value = random.randrange(low, high + 1, step)
            value = min(initials[0] // 2, value) if i > 1 else value
            initials.append(value)
        self.logger.info(f"Topology Optimization - Initial individuals: {initials}")
        return initials

    def mutation(self, individual, indpb):
        """
        Mutation function for topology selection, genetic algorithm.

        Args:
            individual (list): Individual for the genetic algorithm.
            indpb (float): Individual probability for the mutation function, random library criteria.

        Returns:
            list: Mutated individual.
        """
        original = individual[:]
        size = len(individual)
        while True:
            for i in range(size):
                if random.random() < indpb:
                    step = 2 if i > 1 else 10
                    key = list(self.INT_BOUNDS.keys())[i]
                    low, high = self.INT_BOUNDS[key]
                    selectMove = random.choice([-step, step])
                    chosenValue = max(low, min(high, individual[i] + selectMove))
                    individual[i] = chosenValue if i < 2 else min(individual[0] // 2, chosenValue)
            if individual != original:
                break
        return individual

    def customMutate(self, individual, indpb=0.5):
        """
        Mutation function for topology selection, genetic algorithm.

        Args:
            individual (list): Individual for the genetic algorithm.
            indpb (float): Individual probability for the mutation function, random library criteria.

        Returns:
            list: Mutated individual.
        """
        individual = self.mutation(individual, indpb)
        return individual,

    def run(self, populationSize=4, eliteSize=2, maxGenerations=10, maxTime=10000):
        """
        Main function for the genetic algorithm of the topology selection process.

        Args:
            populationSize (int): total number of individual for each generation
            eliteSize (int): number of the best individuals to be stored for the next iteration
            maxGenerations(int): maximum number of iterations in genetic algorithm
            maxTime(int): maximum run time for the genetic algorithm as termination criteria.

        Returns:
            tuple: The best individual and the lowest root mean square errors from the genetic algorithm.
        """
        populations = self.toolbox.population(n=populationSize)
        startTime = time.time()
        currentMinValue = None
        for gen in range(maxGenerations):
            elapsedTime = time.time() - startTime
            if elapsedTime >= maxTime:
                self.logger.info("Topology Optimization - Elapsed time reached, stopping.")
                break
            elites = tools.selBest(populations, eliteSize)
            populations = algorithms.eaSimple(populations, self.toolbox, cxpb=1.0, mutpb=0.5, ngen=1, verbose=False)[0]
            populations.extend(elites)
            populations = tools.selBest(populations, len(populations) - eliteSize)
            fits = [ind.fitness.values[0] for ind in populations]
            self.logger.info(
                f"Topology Optimization - Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            currentMinValue = min(fits)
            self.logger.info(f"Topology Optimization - currentMinValue: {currentMinValue}")
            localIndividual = tools.selBest(populations, 1)[0]
            self.logger.info(f"Topology Optimization - local best parameters: {localIndividual}")
        bestIndividual = tools.selBest(populations, 1)[0]
        self.logger.info(f"Topology Optimization - Best parameters: {bestIndividual}")
        return bestIndividual, currentMinValue
