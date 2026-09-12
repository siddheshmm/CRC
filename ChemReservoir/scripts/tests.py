from scripts.logFileSetting import setupLogger
from scripts.chemReservoir import chemReservoir


loggerShortMemory = setupLogger("shortMemoryTaskLoggerFile")
shortMemoryTask = chemReservoir(1, loggerShortMemory, memoryTask="short", runTime=10, repetition=2, networkOptMaxTime=200)
shortMemoryTask.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
shortMemoryResult=shortMemoryTask.run(maxTime=1000)
print("Short memory results", shortMemoryResult)

loggerLongMemory = setupLogger("longMemoryTaskLoggerFile")
longMemoryTask = chemReservoir(1, loggerLongMemory, memoryTask="long", tau=6, runTime=50, repetition=2, networkOptMaxTime=200)
longMemoryTask.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
longMemoryResult=longMemoryTask.run(maxTime=500)
print("Long memory results", longMemoryResult)

