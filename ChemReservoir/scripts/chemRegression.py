import math
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error



def storeStochSimData(stochDataLength, numberOfMolecules, stochData):
    """
    Storing stochastic simulation data in proper array form separate for the molecule count and time arrays.
    Also not excluding the food molecule data.

    Args:
        stochDataLength (int): length of the array
        numberOfMolecules (int): number of molecules for the array size.
        stochData (np.array): the molecule count from the stochastic simulation.
    Returns:
        tuple: (np.array, np.array) the updated stochastic simulation data.
    """

    concentrations=np.empty((stochDataLength, numberOfMolecules), float)
    stochasticTimeArray= stochData[:,0].astype(float)
    for i in range(0, stochDataLength):
        concentrations[i] = stochData[i][2:].astype(float)
    return stochasticTimeArray, concentrations


def filtering(interpConcentrations):
    """
    Filtering the columns half of which filled by zeros.

    Args:
        interpConcentrations (np.array): interpolated molecule count.

    Returns:
        np.array: the filtering interpolated stochastic simulation data.
    """
    filteringThreshold = int(interpConcentrations.shape[0] * 0.5)
    lessThanOne = interpConcentrations < 1
    count = np.sum(lessThanOne, axis=0)
    filteredColumns = count < filteringThreshold
    newArray = interpConcentrations[:, filteredColumns]
    return newArray


def getShortMemoryTargetData(interpolatedFlow):
    """
    Generating target data for short term memory task.

    Args:
        interpolatedFlow (np.array): interpolated input signal.

    Returns:
        np.array: target data for short term memory task.
    """
    size=len(interpolatedFlow)
    target = np.zeros(size-2)
    for i in range(2, size):
        target[i-2]=interpolatedFlow[i-1]+2*interpolatedFlow[i-2]
    return target


def getLongMemoryTargetData(interpolatedFlow, tau=10):
    """
    Generating target data for long term memory task.

    Args:
        interpolatedFlow (np.array): interpolated input signal.

    Returns:
        np.array: target data for long term memory task.
    """
    size=len(interpolatedFlow)
    target = np.zeros(int(size-int((3/2)*tau)))
    for i in range(int((3/2)*tau), size):
        target[i-int((3/2)*tau)]=interpolatedFlow[i-tau]+(1/2)*interpolatedFlow[i-int((3/2)*tau)]
    return target


def getErrorValues(yTrain, yTrainPred, yTest, yTestPred):
    """
    Calculating normalized root mean square error (NRMSE) value for training and test data.

    Args:
        yTrain (np.array): original training target
        yTrainPred (np.array): predicted training target
        yTest (np.array): original test target
        yTestPred (np.array): predicted test target
    Returns:
        float: NRMSE
    """

    rmseTrain = np.sqrt(mean_squared_error(yTrain, yTrainPred))
    diffTrain=(yTrain.max() - yTrain.min())
    nrmseTrain = rmseTrain / diffTrain if diffTrain!=0 else rmseTrain
    rmseTest = np.sqrt(mean_squared_error(yTest, yTestPred))
    diffTest=(yTest.max() - yTest.min())
    nrmseTest = rmseTest / diffTest if diffTest!=0 else rmseTest
    return nrmseTest


def regressionMemoryTask(trainSplit, sampleValues, interpConcentrations):
    """
    Regression function for memory tasks, short or long term memory.

    Args:
        trainSplit (float): splitting the target data into train and test, default 0.70.
        sampleValues (np.array): target data.
        interpConcentrations (np.array): the interpolated molecule count from the stochastic simulation.

    Returns:
        float: NRMSE
    """

    regression = RidgeCV()
    trainSize = int(len(interpConcentrations) * trainSplit)
    regression.fit(interpConcentrations[0:trainSize], sampleValues[0:trainSize])
    yTrainPred=regression.predict(interpConcentrations[0:trainSize])
    yTestPred=regression.predict(interpConcentrations[trainSize:])
    error= getErrorValues(sampleValues[0:trainSize], yTrainPred, sampleValues[trainSize:], yTestPred[:len(sampleValues[trainSize:])])
    return error


class chemReg:
        
        def __init__(self):
            """
            Initializing the instances for chemRegression class.
            """
            self.experimentalTimes=None
            self.numberOfMolecules=None
            self.modStochSim = None
            self.interpConcentrations=None
            self.trainSplit = None
            self.splitIndices = None
            self.trainIndices = None
            self.testIndices  = None
            self.interpConcentrationsLength=None
            self.concentrationsLength = None
            self.scaledData=None
            self.input=None
            self.sineInput=None
            self.sineInputTime=None
            self.interpolatedFlow=None
            self.firstMoleculeFlow=None
            self.inputFlowTimes=None
        
        def initializeArrays(self, stochData):
            """
            Initializing stochastic simulated related instances variables.
            """

            self.stochDataLength= len(stochData)
            self.numberOfMolecules-=1 #due to additional food molecule

        def interpolationWithExperimentalTimeSeries(self, stochasticTimeArray, concentrations):
            """
            Interpolation with experimental data time series to be able to get the interpolated concentrations
            from the stochsim data.

            Args:
                stochasticTimeArray (np.array): stochastic simulation run times
                concentrations (np.array): molecule counts over time.

            Returns:
                np.array: interpolated molecule counts from stochastic simulation.
            """
            maxTime=int(max(self.inputFlowTimes))
            maxTime+=1
            experimentalTimes = np.arange(0,  maxTime)
            self.getInterpolatedFlow(experimentalTimes)
            self.concentrationsLength=len(experimentalTimes)
            interpConcentrations = np.empty((self.concentrationsLength, self.numberOfMolecules), float)
            count=0
            for i in range(self.numberOfMolecules):
                temp=self.interpolation(
                        experimentalTimes, stochasticTimeArray, concentrations[:,i]
                )
                interpConcentrations[:, count] = temp
                count+=1

            interpConcentrations = filtering(interpConcentrations)
            self.setTrainTestIDX()
            return interpConcentrations

        def preprocessingStochData(self, stochasticTimeArray, concentrations):
            """
            Preprocessing the stochastic simulation data.

            Args:
                stochasticTimeArray (np.array): stochastic simulation run times
                concentrations (np.array): molecule counts over time.

            Returns:
                np.array: preprocessed molecule count.
            """

            self.getInterpolatedFlow(stochasticTimeArray)
            self.concentrationsLength=len(stochasticTimeArray)
            concentrations = filtering(concentrations)
            self.setTrainTestIDX()
            return concentrations

        def setTrainTestIDX(self):
            """
            Setting the test and train index values for the input array before
            the regression step.
            """
            self.splitIndices = math.floor(self.concentrationsLength* self.trainSplit)
            self.trainIndices = [*range(0, self.splitIndices)]
            self.testIndices  = [*range(self.splitIndices, self.concentrationsLength)]
            
        def getInterpolatedFlow(self, stochasticTimeArray):
            """
            The interpolated inflow as the input for basic functions.
            The original target data is already fed with the same input flow as Huck experiments.
            Here, the interpolated flow is used as input
            """
            self.interpolatedFlow=np.interp(stochasticTimeArray, self.inputFlowTimes, self.moleculeFlow)
                
        def interpolation(self, inputXAxis, xAxis, yAxis):
            """
            Interpolation function, used in input signal and stochastic simulation
            preprocessing steps.

            Args:
                inputXAxis (np.array): target input time series
                xAxis (np.array): original time series
                yAxis (np.array): original molecule counts over time.

            Returns:
                np.array: interpolated molecule count.
            """
            return np.interp(inputXAxis, xAxis, yAxis)

        def regressionShortMemoryTargetData(self, interpConcentrations):
            """
            Regression function for short memory task.

            Args:
                interpConcentrations (np.array): interpolated molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            targetData= getShortMemoryTargetData(self.interpolatedFlow)
            error= regressionMemoryTask(self.trainSplit, targetData, interpConcentrations)
            return error

        def regressionLongMemoryTargetData(self, interpConcentrations, tauValue=10):
            """
            Regression function for long memory task.

            Args:
                interpConcentrations (np.array): interpolated molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            delay = int((3 / 2) * tauValue)
            targetData= getLongMemoryTargetData(self.interpolatedFlow, tau=tauValue)
            error= regressionMemoryTask(self.trainSplit, targetData, interpConcentrations[delay:])
            return error

        def runShortMemoryTask(self, stochData, trainSplit=0.7):
            """
            Regression main function for short memory task.

            Args:
                stochData (np.array): molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            self.trainSplit = trainSplit
            self.initializeArrays(stochData)
            stochasticTimeArray, concentrations = storeStochSimData(self.stochDataLength, self.numberOfMolecules, stochData)
            interpConcentrations = self.interpolationWithExperimentalTimeSeries(stochasticTimeArray, concentrations)
            floatError = self.regressionShortMemoryTargetData(interpConcentrations)
            return floatError
            
        def runLongMemoryTask(self, stochData, trainSplit=0.7, tau=10):
            """
            Regression main function for long memory task.

            Args:
                stochData (np.array): molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            self.trainSplit=trainSplit
            self.initializeArrays(stochData)
            stochasticTimeArray, concentrations= storeStochSimData(self.stochDataLength, self.numberOfMolecules, stochData)
            interpConcentrations=self.interpolationWithExperimentalTimeSeries(stochasticTimeArray, concentrations)
            floatError=self.regressionLongMemoryTargetData(interpConcentrations, tauValue=tau)
            return floatError
