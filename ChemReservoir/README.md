# ChemReservoir
ChemReservoir: An Open-Source Framework for Chemically-Inspired Reservoir Computing

Copyright 2025 Mehmet Aziz Yirik

## Introduction

ChemReservoir is an open-source framework for chemically-inspired reservoir modelling. The tool allows the design of optimal chemically-inspired reservoir models based on genetic algorithms.
The combination of Gillespie algorithm based stochastic simulation and regression functions is performed for the evaluation of each individual in the genetic algorithm. The reservoir networks
are generated as cycle-based topologies with chords, local connections to evaluate the impact of the local neighborhood in information processing.

For a detailed explanation and discussion, please refer to the accompanying [ChemReservoir article preprint](https://doi.org/10.48550/arXiv.2506.04249).

## Scripts

The main ChemReservoir framework [script](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts/chemReservoir.py) and all other related scripts are located in [the scripts folder](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts). An example usage of ChemReservoir is given in [tests.py](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts/tests.py). 
The scripts folder consists of out and summary folders, required to be initialized for MØD software to operate properly. The log file for detailed results are given in [results folder](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts/results).

## Dependencies

ChemReservoir utilizes the listed libraries:

- [MØD](https://cheminf.imada.sdu.dk/mod/) (V 0.15.1.17)
- [deap](https://pypi.org/project/deap/) (V 1.4.1)
- [numpy](https://pypi.org/project/numpy/) (V 1.26.4)
- [scipy](https://pypi.org/project/scipy/) (V 1.13.0)


## License
This project is licensed under the MIT License - see [the LICENSE file](https://github.com/MehmetAzizYirik/ChemReservoir/blob/main/LICENSE) file for details.

## Authors

 - Mehmet Aziz Yirik - [MehmetAzizYirik](https://github.com/MehmetAzizYirik) 
