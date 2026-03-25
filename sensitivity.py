#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt
import pysvzerod as zerod
import json
import copy

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

# Need to establish test cases and error tests

# Reads file that is passed as parameter (myfile) and returns the contents of the file
def read_file(myfile):
    with open(myfile, "r") as file:
        data = json.load(file)
    return data

# Returns the 0D element values of ChamberSphere, Upstream_vessel, and 
# Downstream_vessel 0D element types/names given data from a parsed
# file read using read_file function given above 
def get_chamber_params(data):
    vessels = data["vessels"]

    params = None
    down_vals = None
    up_vals = None

    for ves in vessels:
        if (ves["zero_d_element_type"] == "ChamberSphere"):
            params = ves["zero_d_element_values"]

        elif (ves["vessel_name"] == "downstream_vessel"):
            down_vals = ves["zero_d_element_values"]
        
        elif (ves["vessel_name"] == "upstream_vessel"):
            up_vals = ves["zero_d_element_values"]

    return params, down_vals, up_vals

# Returns the resistance value of the given vessel name, if possible
# Otherwise returns error
def get_res(data, ves_name):
    for ves in data["vessels"]:

        if ves["vessel_name"] == ves_name:
            vals = ves["zero_d_element_values"]

            if "R_poiseuille" in vals:
                return vals["R_poiseuille"]
            else:
               raise LookupError("Error: Given vessel type does not contain resistance parameter")
            
    raise LookupError(f"Vessel '{ves_name}' not found in data.")

def change_res(data, scaler, ves_name):
    perturbed_res = scaler*get_res(data, ves_name)
    for ves in data["vessels"]:
        if ves["vessel_name"] == ves_name:
            ves["zero_d_element_values"]["R_poiseuille"] = perturbed_res

def get_p_metric(data, metric):
    results = zerod.simulate(data)
    pressure = np.array(results[results.name == "pressure:outlet_valve:downstream_vessel"].y) # Accesses LV Pressure
    if(metric == "max"):
        p_max = np.max(pressure)
        return p_max
    
    elif(metric == "min"):
        p_min = np.min(pressure)
        return p_min
    
    elif(metric == "mean"):
        p_mean = np.mean(pressure)
        return p_mean




def create_pmax_res_graph(data, ves_name, num_pts):
    if(num_pts<=2):
        raise ValueError("Need at least 3 points for array")

    scalers = np.linspace(0.5, 2, num_pts)

    resistances = np.zeros(num_pts)
    p_max_values = np.zeros(num_pts)
    p_mean_values = np.zeros(num_pts)


    for i in range(num_pts):
        perturbed_data = copy.deepcopy(data)
        change_res(perturbed_data, scalers[i], ves_name)
        resistances[i] = get_res(perturbed_data, ves_name)

        p_max_values[i] = get_p_metric(perturbed_data,"max")
        p_mean_values[i] = get_p_metric(perturbed_data,"mean")

        

    
    plt.plot(resistances, p_max_values, label = "Max pressures")
    plt.plot(resistances, p_mean_values, label = "Mean pressures")

    plt.title(f"Pressure metrics vs Resistances ({ves_name})")
    plt.xlabel("Resistance")
    plt.ylabel("Pressure")
    plt.legend()
    plt.show()

    unperturbed_res = get_res(data, ves_name)
    print(f"- Baseline resistance:\n\t{unperturbed_res}\n")
    print(f"- Baseline max pressure:\n\t{get_p_metric(data, 'max')}\n")
    print(f"- Baseline mean pressure:\n\t{get_p_metric(data, 'mean')}\n")
    
    print(f"- Resistances computed:\n\t{resistances}\n")
    print(f"- Max pressures computed:\n\t{p_max_values}\n")
    print(f"- Mean pressures computed:\n\t{p_mean_values}\n")
    
    
    # Statisitics
    print(f"- Standard deviation of max pressure values:\n\t{np.std(p_max_values)}\n")
    print(f"- Standard deviation of mean pressure values:\n\t{np.std(p_mean_values)}\n")


    p_max_res_sens = np.gradient(p_max_values, resistances)
    norm_p_max_res_sens = (p_max_res_sens*resistances)/p_max_values
    print(f"- Normalized sensitivty (max pressure, resistance):\n\t{norm_p_max_res_sens}\n")

    p_mean_res_sens = np.gradient(p_mean_values, resistances)
    norm_p_mean_res_sens = (p_mean_res_sens*resistances)/p_mean_values
    print(f"- Normalized sensitivty (mean pressure, resistance):\n\t{norm_p_mean_res_sens}\n")



def change_cap(data, scaler, ves_name):
    for ves in data["vessels"]:
        if ves["vessel_name"] == ves_name:
            vals = ves["zero_d_element_values"]

            if "C" in vals:
                vals["C"] *= scaler


# Function that measures sensitivity of pressure (could generalize to allow user to decide - create parameter to allow this eventually) 
# through the sobol variance-based method (as used and slightly detailed Saltelli et al. (2010)).
# Need to estimate "sensitivity indices" of the "first-order, second-order, and total effect indices." (requires emulator...)
def sobol_sensitivity(data,ves_name):
    #first order sensitivity coefficient is given as
    # sensitivity_i = \frac{Variance_{X_i}(E_{X_\sim i}(Y | X_i)}{Variance(Y)}

    # total effect index is given as
    # sensitivity_{T_i} = 1-\frac{Variance_{X_i}(E_{X_\sim i}(Y | X_i)}{Variance(Y)}
    # which measures the total effect, i.e. first and higer order effects (interactions) of
    # factor X_i. 

    placeholder = 1

####
# Below are the SALib-based functions
####

# Parameters:
# 
def evaluate_model(data, params):
    Y = np.zeros(len(params))

    # index i, element X (which is a singular row of column vector params)
    for i, X in enumerate(params):
        perturbed_data = copy.deepcopy(data)

        # Getting the following error: RuntimeError: Maximum number of non-linear iterations reached.
        # Thus, failing due to zerodsolver as opposed to SALib, meaning parameters causing
        # error. DO try/except and then analyze where the simulation is failing.
        # Ask insight from Martin on why he thinks solver is failing. Is it that the 
        # parameters are so physically impropable that the values just cause the solver
        # to crash?
        # Tried the above, which allowed sobol to run successfully, but then some of
        # the sobol SA outputs were NaN. Thus, will just reduce the range of the
        # bounds from 0.5 - 2.0 to 0.8 - 1.2 and see how that works
        Up_res_scaler = X[0]  
        Down_res_scaler = X[1]
        Up_cap_scaler = X[2]
        Down_cap_scaler = X[3]




        change_res(perturbed_data, Up_res_scaler, "upstream_vessel")
        change_res(perturbed_data, Down_res_scaler, "downstream_vessel")
        change_cap(perturbed_data, Up_cap_scaler, "upstream_vessel")
        change_res(perturbed_data, Down_cap_scaler, "downstream_vessel")

        Y[i] = get_p_metric(perturbed_data, "max") #computing Y = f(X) essentially
    
    # except:
    #     print(f"Zerod simulation failed at index i = {i}, given params: {X}")
    #     Y[i] = np.nan
    
    # # Debugging to ensure size mistmatch isn't cause for indexing error
    # print("param_values shape:", params.shape)
    # print("Y shape:", Y.shape)

    return Y

def sobol_sensitivity(data):
    problem = {
        'num_vars': 4,
        'names': ['Up_res', "Down_res", "Up_cap", 'Down_cap'],
        'bounds': [[0.8, 1.2]]*4 # note these are arbitrarily chosen, not physically grounded - not sure if the range is too small
    }

    # Using 512 as arbitrary sample size (for greater accuracy, could 
    # double or quadurple sample size). Want to keep sample size power of 2
    # This step generates the input (sobol) parameter set
    # Total # of model evaluations will be N (2*NV+2), where NV is number of variables (given NV>1)
    params = sobol_sample.sample(problem, 512, calc_second_order=True) 

    Y = evaluate_model(data, params)

    # Provides insight into how much each parameter contributes to output variance
    # Essentially the actual evaluation step where the sobol variance-based
    # sensitivty analysis formula described in Stelli is evaluated.
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=True)

    return Si

####################
# End of functions #
####################


fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/svZeroDSolver-jt/tests/cases/chamber_sphere.json"

baseline_input = read_file(fname)
baseline_input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
baseline_input["simulation_parameters"]["output_all_cycles"] = False

# pdb.set_trace() # review documentation (placed here in order to analyze baseline_
                #results.name)

problem = {
    'num_vars': 4, 
    'names': ['Up_res', "Down_res", "Up_cap", 'Down_cap'],
    'bounds': [[0.8, 1.2]]*4 # note these are arbitrarily chosen, not physically grounded - not sure if the range is too small
}

params = sobol_sample.sample(problem, 512)


Si = sobol_sensitivity(baseline_input)

