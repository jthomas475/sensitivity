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


def create_dict(json_file):
    param_dict = {}

    for ves in json_file["vessels"]:
        name = ves["vessel_name"]
        params = {}

        for k, v in ves["zero_d_element_values"].items():
            if isinstance(v, (int, float)):
                params[k] = v

        if params:
            param_dict[name] = {
                "type": "vessel",
                "params": params
            }

    for valve in json_file["valves"]:
        name = valve["name"]
        params = {}

        for k, v in valve["params"].items():
            if isinstance(v, (int, float)):
                params[k] = v
        if params:
            param_dict[name] = {
                "type": "valve",
                "params": params
            }

    return param_dict

def create_indicators(param_dict):
    param_indicators = []
    param_map = []

    for ves_name, block in param_dict.items():
        for param_name in block["params"]:
            ves_param_name = f"{ves_name}-{param_name}"

            param_indicators.append(ves_param_name)

            param_map.append({
                "type": block["type"],
                "name": ves_name,
                "param": param_name
            })


    return param_indicators, param_map
    
####
# Below are the SALib-based functions
####

# Parameters:
# 
def evaluate_model(data, params, param_map, metric):
    Y = np.zeros(len(params))
    failures = 1

    # index i, element X (which is a singular row of column vector params)
    for i, X in enumerate(params):
        perturbed_data = copy.deepcopy(data)

        failed_params = []

        for j, val in enumerate(param_map):
            ves = val["name"]
            param = val["param"]
            scaler = X[j]

            failed_params.append((ves, param,scaler))

            if val["type"] == "vessel":
                for vessel in perturbed_data["vessels"]:
                    if vessel["vessel_name"]== ves:
                        vessel["zero_d_element_values"][param] *= scaler

            elif val["type"] == "valve":
                for valve in perturbed_data["valves"]:
                    if valve["name"] == ves:
                        valve["params"][param] *= scaler

        try:
            Y[i] = get_p_metric(perturbed_data, metric)

        except RuntimeError as e:
            print(f"Failure numer: {failures}")
            print(f"Simulation failed at index {i}")
            print("Parameters used:")
            for ves, param, scaler in failed_params:
                print(f" {ves}-{param}: {scaler}")
            print(f"Error message: {e}")
            failures+=1
            Y[i] = get_p_metric(data, metric)

    return Y


# Function that measures sensitivity of pressure (could generalize to allow user to decide - create parameter to allow this eventually) 
# through the sobol variance-based method (as used and slightly detailed Saltelli et al. (2010)).
# Need to estimate "sensitivity indices" of the "first-order, second-order, and total effect indices." (requires emulator...)
#first order sensitivity coefficient is given as
# sensitivity_i = \frac{Variance_{X_i}(E_{X_\sim i}(Y | X_i)}{Variance(Y)}
# total effect index is given as
# sensitivity_{T_i} = 1-\frac{Variance_{X_i}(E_{X_\sim i}(Y | X_i)}{Variance(Y)}
# which measures the total effect, i.e. first and higer order effects (interactions) of
# factor X_i. 

#Instead of providing an array of strings containing parameter names/acronyms,
# establish a dictionary called param_dict  
def sobol_sensitivity(data, param_dict, bound, metric):    
    param_indicators, param_map = create_indicators(param_dict)

    num_vars = len(param_indicators)

    problem = {
        'num_vars': num_vars,
        'names': param_indicators,
        'bounds': [bound]*num_vars # note these are arbitrarily chosen, not physically grounded - not sure if the range is too small
    }

    # Using 512 as arbitrary sample size (for greater accuracy, could 
    # double or quadurple sample size). Want to keep sample size power of 2
    # This step generates the input (sobol) parameter set
    # Total # of model evaluations will be N (2*NV+2), where NV is number of variables (given NV>1)
    params = sobol_sample.sample(problem, 512, calc_second_order=True) 

    Y = evaluate_model(data, params, param_map, metric)

    # Provides insight into how much each parameter contributes to output variance
    # Essentially the actual evaluation step where the sobol variance-based
    # sensitivty analysis formula described in Stelli is evaluated.
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=True)

    return Si


####################
# Heatmap Function #
####################
def sobol_heatmap(Si, index, param_dict):
    param_indicators, param_map = create_indicators(param_dict)

    Si_index_matrix = np.abs(Si[index])
         
    plt.figure(figsize=(10,8))
    plt.imshow(Si_index_matrix)

    plt.xticks(range(len(param_indicators)), param_indicators, rotation=90)
    plt.yticks(range(len(param_indicators)), param_indicators)

    plt.colorbar(label=f"{index}")
    plt.title(f"Sobol {index} sensitivity analysis heatmap - max pressure")

    plt.tight_layout()
    plt.show()

####################
# End of functions #
####################


LV_fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/svZeroDSolver-jt/tests/cases/chamber_sphere.json"

LV_baseline_input = read_file(LV_fname)
LV_baseline_input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
LV_baseline_input["simulation_parameters"]["output_all_cycles"] = False



Caruel_fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/sensitivity/chamber_sphere_Caruel.json"

Caruel_baseline_input = read_file(Caruel_fname)
Caruel_baseline_input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
Caruel_baseline_input["simulation_parameters"]["output_all_cycles"] = False

# pdb.set_trace() # review documentation (placed here in order to analyze baseline_
                #results.name)

bound = [0.8,1.2]
# LV_problem = {
#     'num_vars': 4, 
#     'names': ['Up_res', "Down_res", "Up_cap", 'Down_cap'],
#     'bounds': [[0.8, 1.2]]*4 # note these are arbitrarily chosen, not physically grounded - not sure if the range is too small
# }

# Caruel_problem = {
#     'num_vars': 6, 
#     'names': ['Up_res', "Down_prox_res","Down_dist_res", "Up_cap", 'Down_prox_cap',"Down_dist_cap"],
#     'bounds': [[0.8, 1.2]]*6
# }

LV_dict = create_dict(LV_baseline_input)

Caruel_dict = create_dict(Caruel_baseline_input)

Si_LV = sobol_sensitivity(LV_baseline_input, LV_dict, bound, "max")

# Si_Caruel = sobol_sensitivity(Caruel_baseline_input, Caruel_dict, bound, "max")

sobol_heatmap(Si_LV, "S2", LV_dict)
# heatmap(Si_LV, "S1", LV_dict)
# heatmap(Si_LV, "ST", LV_dict)
print(f"S2 values:\n {Si_LV[S2]}")