#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt
import pysvzerod as zerod
import json
import copy
import time

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from SALib import ProblemSpec

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
    # pressure = np.array(results[results.name == "pressure:outlet_valve:downstream_proximal_vessel"].y) # Accesses Caruel Pressure for proximal vessel
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
    

def get_radius(data):
    for ves in data["vessels"]:
        if ves["zero_d_element_type"]=="ChamberSphere":
            return ves["zero_d_element_values"]["radius0"]

def get_v_metric(data, metric):
    results = zerod.simulate(data)

    LV_radius = get_radius(data)

    vol = np.array(results[results.name == "volume:ventricle"].y) + (4/3)*np.pi*LV_radius**3 # need to fix - volume is volume change (calculate baseline volume)
    time = np.array(results[results.name == "volume:ventricle"].time)

    if(metric == "max"):
        v_max = np.max(vol)
        return v_max
    
    elif(metric == "min"):
        v_min = np.min(vol)
        return v_min
    
    elif(metric == "mean"):
        v_mean = np.mean(vol)
        return v_mean



def get_f_metric(data, metric):
    results = zerod.simulate(data)
    flow = np.array(results[results.name == "flow:outlet_valve:downstream_vessel"].y) # Accesses LV Pressure
    if(metric == "max"):
        f_max = np.max(flow)
        return f_max
    
    elif(metric == "min"):
        f_min = np.min(flow)
        return f_min
    
    elif(metric == "mean"):
        f_mean = np.mean(flow)
        return f_mean

# Note that end-diastolic volume (EDV) is 100% of the max volume (so just max volume)
# End-systolic volume (ESV) is x% of the max volume (about 30%-40% max volume)
# stroke volume is EDV-ESV
# Ejection fraction is stroke Volume / EDV * 100
def get_outputs(data):
    EDV = get_v_metric(data,"max")

    ESV = get_v_metric(data,"min")

    stroke = EDV-ESV

    EF = stroke/EDV *100

    return EDV, ESV, stroke, EF



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

    # for valve in json_file["valves"]:
    #     name = valve["name"]
    #     params = {}

    #     for k, v in valve["params"].items():
    #         if isinstance(v, (int, float)):
    #             params[k] = v
    #     if params:
    #         param_dict[name] = {
    #             "type": "valve",
    #             "params": params
    #         }

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
    p_maxs, EDV, ESV, stroke, EF = [np.full(len(params), np.nan) for _ in range(5)]
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
            p_maxs[i] = get_p_metric(perturbed_data, metric)
            EDV[i], ESV[i], stroke[i], EF[i] = get_outputs(perturbed_data)


        except RuntimeError as e:
            print(f"Failure numer: {failures}")
            print(f"Simulation failed at index {i}")
            print("Parameters used:")
            for ves, param, scaler in failed_params:
                print(f" {ves}-{param}: {scaler}")
            print(f"Error message: {e}")

            failures+=1

    return p_maxs, EDV, ESV, stroke, EF


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
        'bounds': [bound]*num_vars, # note these are arbitrarily chosen, not physically grounded - not sure if the range is too small
        'outputs': ["p_maxs","EDV","ESV","stroke","EF"]
    }

    # Using 512 as arbitrary sample size (for greater accuracy, could 
    # double or quadurple sample size). Want to keep sample size power of 2
    # This step generates the set of model inputs
    # Total # of model evaluations will be N (2*NV+2), where NV is number of variables (given NV>1)
    params = sobol_sample.sample(problem, 1024, calc_second_order=True) 

    # Keep track of the elapsed time of the program 
    startTime = time.time()

    p_maxs, EDV, ESV, stroke, EF = evaluate_model(data, params, param_map, metric)

    np.save("./raw_outputs.npy", np.stack([p_maxs, EDV, ESV, stroke, EF]))
    print("Raw outputs saved.")

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)

    print(f"It took {int(mins)} minutes and {secs:.1f} seconds to evaluate the model. ({elapsedTime} seconds total)")

    # only save non-null rows, otherwise omit 
    valid_rows = ~(np.isnan(p_maxs) | np.isnan(EDV) | np.isnan(ESV) | np.isnan(stroke) | np.isnan(EF))
    # clean_p_maxs = p_maxs[valid_rows]
    # clean_EDV = EDV[valid_rows]
    # clean_ESV = ESV[valid_rows]
    # clean_stroke = stroke[valid_rows]
    # clean_EF = EF[valid_rows]


    # Quantify how many rows contained NaN values - used to help determine whether enough valid results were quantified
    failure_rate = (~valid_rows).sum() / len(valid_rows)
    print(f"Failure rate: {failure_rate}")


    if failure_rate > 0.05:
        print("WARNING: Failure rate > 0.05. Sensitivity indices may be unreliable.")


    for arr in [p_maxs, EDV, ESV, stroke, EF]:
        nanMask = np.isnan(arr) # indicate which rows/values of the array are NaN (boolean mask: true if NaN)
        if nanMask.any():
            arr[nanMask] = np.nanmedian(arr) # for all rows that are nan replace nan value with median of all valid rows

    # Provides insight into how much each parameter contributes to output variance
    # Essentially the actual evaluation step where the sobol variance-based
    # sensitivty analysis formula described in Stelli is evaluated.
    Si_p_maxs = sobol.analyze(problem, p_maxs, calc_second_order=True, print_to_console=True)
    Si_EDV = sobol.analyze(problem, EDV, calc_second_order=True, print_to_console=True)
    Si_ESV = sobol.analyze(problem, ESV, calc_second_order=True, print_to_console=True)
    Si_stroke = sobol.analyze(problem, stroke, calc_second_order=True, print_to_console=True)
    Si_EF = sobol.analyze(problem, EF, calc_second_order=True, print_to_console=True)
    
    return Si_p_maxs, Si_EDV, Si_ESV, Si_stroke, Si_EF, problem["outputs"]


####################
# Heatmap Function #
####################
def s2_heatmap(Si, param_dict):
    param_indicators, param_map = create_indicators(param_dict)

    index = "S2"
    Si_index_matrix = np.abs(Si[index])
         
    plt.figure(figsize=(10,8))
    plt.imshow(Si_index_matrix)

    plt.xticks(range(len(param_indicators)), param_indicators, rotation=90)
    plt.yticks(range(len(param_indicators)), param_indicators)

    plt.colorbar(label=f"{index}")
    plt.title(f"Sobol {index} sensitivity analysis heatmap - max pressure")

    plt.tight_layout()
    plt.show()

# @param Si_s: Array of sobol analysis results
# @ param: names: The names of the outputs
# @param param_dict: dictionary of the parameters
# @param names: array of sensitivty analysis value/output names (["Y","EDV","ESV","stroke","EF"])
def SA_heatmap(Si_s, names, param_dict):
    #pdb.set_trace()

    param_indicators, param_map = create_indicators(param_dict)
    indices = ["ST","S1"]

    mymap = []

    for sobol_idx in indices:
        for i, Si in enumerate(Si_s):
            Si_index_matrix = np.array(Si[sobol_idx]) # an array containing all the sobol values for given sobol_idx

            Si_index_matrix[Si_index_matrix < 0] = 0


            mymap.append(Si_index_matrix)
    

    half = int(len(mymap) / 2) 
    heatmap_ST = np.array(mymap[:half]).T
    heatmap_SI = np.array(mymap[half:]).T

    # Print heatmap for ST
    plt.figure(figsize=(10,8))
    plt.imshow(heatmap_ST)

    plt.yticks(range(len(param_indicators)), param_indicators)
    plt.xticks(range(len(names)), names, rotation=90)

    plt.colorbar(label=f"ST")
    plt.title(f"Sobol ST sensitivity analysis heatmap")

    plt.tight_layout()
    plt.show()   

    # Print heatmap for SI
    plt.figure(figsize=(10,8))
    plt.imshow(heatmap_SI)

    plt.yticks(range(len(param_indicators)), param_indicators)
    plt.xticks(range(len(names)), names, rotation=90)

    plt.colorbar(label=f"SI")
    plt.title(f"Sobol SI sensitivity analysis heatmap")

    plt.tight_layout()
    plt.show()   

####################
# End of functions #
####################


LV_fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/sensitivity/chamber_sphere.json"

LV_baseline_input = read_file(LV_fname)
# LV_baseline_input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
LV_baseline_input["simulation_parameters"]["output_all_cycles"] = False



# Caruel_fname = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/sensitivity/chamber_sphere_Caruel.json"

# Caruel_baseline_input = read_file(Caruel_fname)
# Caruel_baseline_input["simulation_parameters"]["number_of_cardiac_cycles"] = 2
# Caruel_baseline_input["simulation_parameters"]["output_all_cycles"] = False

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

# Caruel_dict = create_dict(Caruel_baseline_input)

Si_y, Si_EDV, Si_ESV, Si_stroke, Si_EF, output_names = sobol_sensitivity(LV_baseline_input, LV_dict, bound, "max")

# #Si_Caruel = sobol_sensitivity(Caruel_baseline_input, Caruel_dict, bound, "max")

LV_SIs = [Si_y, Si_EDV, Si_ESV, Si_stroke, Si_EF]

# np.save("./LV_SIs",LV_SIs)
np.save("./LV_SIs_1024_5/18/2026",LV_SIs)
#LV_SIs = np.load("./LV_SIs.npy", allow_pickle=True)

SA_heatmap(LV_SIs, ["Max Pressure","EDV","ESV","STROKE","EF"], LV_dict)
# # print(f"S2 values:\n {Si_LV[S2]}")

