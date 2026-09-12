import mod
from mod import *
import numpy as np
from scripts.chemSimulation import simFunc
from scripts.chemRegression import chemReg
from scipy.stats import truncnorm

class shortMemoryTask:
    def __init__(self, seed, dg, moleculeInflux, arguments, runTime=50, repetition=2):
        """
        Setting instance variables for the short memory task.

        Args:
            seed (int): seed value for the random and np.random libraries
            dg (tuple): (molecules, rules) used for the construction of derivation graph in MOD.
            moleculeInflux (int): the fixed molecule amount to be used in inflow signal generation.
            arguments (list): input parameters for the simulation.
            runTime (int): simulation run time
            repetition (int): repetition value in input signal for the input hold time.
        """

        np.random.seed(seed)
        self.DG = dg
        self.runTime=runTime
        self.repetition=repetition
        self.moleculeInflux=moleculeInflux
        self.Food1 = self.DG[0][0]
        self.arguments=arguments

    @staticmethod
    def truncated_normal_distribution(moleculeInflux, size, repetition=5):
        """
        Truncated normal distribution for input signal (molecule influx values) generation.

        Args:
           moleculeInflux (int): the fixed molecule amount to be used in inflow signal generation.
           size (int): the length of the input signal.
           repetition (int): repetition value in input signal for the input hold time.

        Returns:
            np.array: input signal, molecule influx values with fixed input hold time value.
        """
        size = int(size / repetition)
        mu, sigma, lower, upper = 0.5, 0.25, 0, 1
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        truncated_normal = truncnorm(a, b, loc=mu, scale=sigma).rvs(size)*moleculeInflux
        return np.repeat(np.array(truncated_normal), repetition)

    def main(self):
        """
        Main function for short term memory task. normal distribution for input signal (molecule influx values) generation.

        Returns:
            float: the lowest normalized root mean square error from network optimization process.
        """

        input_graphs, input_rules = self.DG[0], self.DG[1]
        sim = simFunc(self.arguments)
        sim.Food1=self.Food1
        cReg =  chemReg()
        sim.inputGraphs, sim.inputRules = input_graphs, input_rules
        sim.reactionRatesDict = {input_rules[i]: float(rate)*sim.reactionRateScale for i, rate in enumerate(sim.reactionRatesList)}
        sim.numberOfMolecules = len(input_graphs)
        sim.numberOfReactions = len(input_rules)
        sim.initializedMolecules = [self.Food1]
        sim.initializeIRate()
        sim.initializeMoleculeLabels()
        initialStates = {self.Food1: 0}
        sim.totalRunTime = self.runTime

        inflow = shortMemoryTask.truncated_normal_distribution(self.moleculeInflux, sim.totalRunTime, repetition=self.repetition)
        sim.generateRates(inflow, rStart=10, rEnd=100)
        traces=sim.simulation(input_rules, input_graphs, initialStates)
        matrix = np.array(sim.moleculeFlow)
        cReg.scalingRate = sim.scalingRate
        cReg.moleculeFlow = matrix
        cReg.inputFlowTimes = sim.inputFlowTimes
        cReg.numberOfMolecules = sim.numberOfMolecules
        stochData = sim.retrieveConcentrationInfoOverTime(traces, sim.numberOfMolecules, sim.totalRunTime)
        error = cReg.runShortMemoryTask(stochData)
        return error
