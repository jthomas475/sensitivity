#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt
import pysvzerod as zerod
import json


def read_file(myfile):
    with open(myfile, "r") as file:
        data = json.load(file)
    return data

fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/svZeroDSolver-jt/tests/cases/chamber_sphere.json"

input = read_file(fname)
input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
input["simulation_parameters"]["output_all_cycles"] = False
input_reference = input.copy()

results = zerod.simulate(input)

pressure = np.array(results[results.name == "pressure:outlet_valve:downstream_vessel"].y)
p_max = np.max(pressure)

plt.plot(pressure)
plt.show()

pdb.set_trace()

