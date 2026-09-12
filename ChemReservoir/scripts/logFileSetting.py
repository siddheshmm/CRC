import logging
import os

def setupLogger(fileName, dir="results"):
    """
    Setting the loger file for the intermediate steps of the chemReservoir tool.

    Args:
        fileName (string): title of the log file
        dir (string): where the file is stored

    Returns:
        logging.Logger: logger file for intermediate steps.
    """
    os.makedirs(dir, exist_ok=True)

    logger = logging.getLogger(fileName)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logPath = os.path.join(dir, f"{fileName}.log")
        fileHandler = logging.FileHandler(logPath)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger
