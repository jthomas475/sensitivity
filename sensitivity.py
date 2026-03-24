#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt
import pysvzerod as zerod
import json
import copy

# Need to establish test cases and error tests

def read_file(myfile):
    with open(myfile, "r") as file:
        data = json.load(file)
    return data


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

def get_res(data, ves_name):
    for ves in data["vessels"]:

        if ves["vessel_name"] == ves_name:
            vals = ves["zero_d_element_values"]

            if "R_poiseuille" in vals:
                return vals["R_poiseuille"]
            else:
                print("Error: Given vessel type does not contain resistance parameter")

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


####################
# End of functions #
####################


fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/svZeroDSolver-jt/tests/cases/chamber_sphere.json"

baseline_input = read_file(fname)
baseline_input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
baseline_input["simulation_parameters"]["output_all_cycles"] = False

perturbed_input = copy.deepcopy(baseline_input)

baseline_results = zerod.simulate(baseline_input)

baseline_pressure = np.array(baseline_results[baseline_results.name == "pressure:outlet_valve:downstream_vessel"].y) # Accesses LV Pressure

# pdb.set_trace() # review documentation (placed here in order to analyze baseline_
                #results.name)


p_max = np.max(baseline_pressure)

plt.plot(baseline_pressure, label="Baseline")


cur_params, cur_down_vals, cur_up_vals = get_chamber_params(baseline_input)


# Perform sensitivity analysis on upstream/downstream vessel resistance for given file
change_res(perturbed_input, 1.15, "downstream_vessel")

down_res_results = zerod.simulate(perturbed_input)

perturbed_pressure = np.array(down_res_results[down_res_results.name == "pressure:outlet_valve:downstream_vessel"].y)
p_max = np.max(perturbed_pressure)

plt.plot(perturbed_pressure, label="Perturbed (resistance*1.15)")
plt.title("Pressure vs Time")
plt.legend()
plt.show()



create_pmax_res_graph(baseline_input, "downstream_vessel", 10)


# Need to check how sensitivity is calculated in paper
# Need to evaluate upstream_vals

