#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import pysvzerod as zerod
import json
import copy
import time
import os
from joblib import Parallel, delayed
from contextlib import contextmanager
from tqdm import tqdm


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

def safe_save(path, data):
    if not path.endswith(".npy"):
        path = path + ".npy"

    if os.path.exists(path):
        raise FileExistsError(f"File '{path}' already exists. Update file name to avoid overwriting data")
    
    np.save(path, data)
    print(f"Succesfully saved path to '{path}'")

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

def get_closed_loop_params(data):
    vessels = data[vessels]

    expmat_vals = None
    aorta_vals = None
    arteries_vals = None
    arterioles_vals = None
    caps_vals = None
    venueles_vals = None
    veins_vals = None

    for ves in vessels:
        if (ves["vessel_name"] == "LV"):
            expmat_vals = ves["zero_d_element_values"]

        elif (ves["vessel_name"] == "aorta"):
            aorta_vals = ves["zero_d_element_values"]

        elif (ves["vessel_name"] == "arteries"):
            arteries_vals = ves["zero_d_element_values"]
        
        elif (ves["vessel_name"] == "arterioles"):
            arterioles_vals = ves["zero_d_element_values"]

        elif (ves["vessel_name"] == "capillaries"):
            caps_vals = ves["zero_d_element_values"]
        
        elif (ves["vessel_name"] == "venules"):
            venueles_vals = ves["zero_d_element_values"]

        elif (ves["vessel_name"] == "veins"):
            veins_vals = ves["zero_d_element_values"]

    return expmat_vals, aorta_vals, arteries_vals, arterioles_vals, caps_vals, venueles_vals, veins_vals

# Obtain the y-value for a given variable/name of the model
# Runtime error is raised if variable is not found, which is recorded as a failure
# instead of crashing when empty array observed.
def extract_val(results, name):
    y = np.array(results[results.name == name].y)
    if y.size == 0:
        raise RuntimeError(f"No '{name}' rows in results (solve diverged or variable absent)")
    return y
    

def get_radius(data):
    for ves in data["vessels"]:
        if ves["zero_d_element_type"]=="ChamberSphere":
            return ves["zero_d_element_values"]["radius0"]



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


#Contextmanager implementation to implement verbose/quiet toggle
# Utilize that every running program has file descriptors, with
# 0 representing stdin, 1 stdout, 2 stdin
@contextmanager
def quiet_func():
    devnull = os.open(os.devnull, os.O_WRONLY) # open null device and write only
    saved = os.dup(1) # save current stdout to record where stdout is written if 1 file pipeline disrupted
    os.dup2(devnull,1) # make 1 (stdout) point to wherever devnull points
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved) # close the temp descriptor and null point
        os.close(devnull)

####
# Below are the SALib-based functions
####


def evaluate_single(data, params_row, param_map, metric, model):
    perturbed_data = copy.deepcopy(data)

    for i, X in enumerate(param_map):
        ves = X["name"]
        param = X["param"]
        scaler = params_row[i]

        if X["type"] == "vessel":
            for vessel in perturbed_data["vessels"]:
                if vessel["vessel_name"]== ves:
                    vessel["zero_d_element_values"][param] *= scaler
        
        elif X["type"] == "valve":
            for valve in perturbed_data["valves"]:
                if valve['names']== ves:
                    valve["zero_d_element_values"][param] *= scaler
    
    try:
        return evaluate_sample(perturbed_data, metric, model), None

    except (RuntimeError, ValueError) as e:
        return (np.nan, np.nan, np.nan, np.nan, np.nan), str(e)



# Note that end-diastolic volume (EDV) is 100% of the max volume (so just max volume)
# End-systolic volume (ESV) is x% of the max volume (about 30%-40% max volume) - just defaulted to obtaining the min vol value
# stroke volume is the difference between EDV and ESV. Formally, EDV-ESV
# Ejection fraction (EF) is stroke Volume / EDV * 100
# Evaluate sample once as opposed to three times to improve SA efficiency
def evaluate_sample(data, metric, model="open"):
    with quiet_func():
        results = zerod.simulate(data)

    p_name = "pressure:outlet_valve:downstream_vessel" if model == "open" else "pressure:LV:AV"
    v_name = "volume:ventricle" if model == "open" else "volume:LV"

    pressure = extract_val(results, p_name)
    if metric == "max":
        p_metric = np.max(pressure)
    elif metric == "min":
        p_metric = np.min(pressure)
    elif metric == "mean":
        p_metric = np.mean(pressure)
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    vol = extract_val(results, v_name)
    if model == "open":
        # volume:ventricle is the change in volume, so add the baseline sphere volume
        vol = vol + (4/3)*np.pi*get_radius(data)**3

    EDV = np.max(vol)
    ESV = np.min(vol)
    stroke = EDV - ESV
    EF = stroke / EDV * 100

    return p_metric, EDV, ESV, stroke, EF


# running parallel with n_jobs and default backend module "loky" to "start separate Python worker processes to execute tasks concurrently on separate CPUs." 
def evaluate_model_parallel(data, params, param_map, metric, model="open", n_jobs=1):
    p_maxs, EDV, ESV, stroke, EF = [np.full(len(params), np.nan) for _ in range(5)]
    failures = 0

    results = Parallel(n_jobs=n_jobs, return_as = "generator")(
        delayed(evaluate_single)(data, params[i], param_map, metric, model) for i in range(len(params))
    ) 

    # note e represents error
    for i, (vals, e) in enumerate(results):
        p_maxs[i], EDV[i], ESV[i], stroke[i], EF[i] = vals

        if e is not None:
            failures += 1
            for val in param_map:
                print(f"[fail {failures}] sample {i}: {val['name']}-{val['param']}: {params[i][param_map.index(val)]}")
                print(f"Error message: {e}")

    return p_maxs, EDV, ESV, stroke, EF




def evaluate_model(data, params, param_map, metric, model="open"):
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
            p_maxs[i], EDV[i], ESV[i], stroke[i], EF[i] = evaluate_sample(perturbed_data, metric, model)


        except (RuntimeError, ValueError) as e:
            print(f"Failure number: {failures}")
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
def sobol_sensitivity(data, param_dict, bound, metric, model, n_jobs=1):    
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
    params = sobol_sample.sample(problem, 16, calc_second_order=True) 

    # Keep track of the elapsed time of the program 
    startTime = time.time()

    # p_maxs, EDV, ESV, stroke, EF = evaluate_model(data, params, param_map, metric, model)
    p_maxs, EDV, ESV, stroke, EF = evaluate_model_parallel(data, params, param_map, metric, model, n_jobs)

    # safe_save("./RawOutputs/raw_outputs_6_2_2026_closedloop_samplesize16_bound0.4_test1.npy", np.stack([p_maxs, EDV, ESV, stroke, EF]))
    # print("Raw outputs saved.")

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
    # sensitivty analysis formula described in Satelli is evaluated.
    Si_p_maxs = sobol.analyze(problem, p_maxs, calc_second_order=True, print_to_console=True)
    Si_EDV = sobol.analyze(problem, EDV, calc_second_order=True, print_to_console=True)
    Si_ESV = sobol.analyze(problem, ESV, calc_second_order=True, print_to_console=True)
    Si_stroke = sobol.analyze(problem, stroke, calc_second_order=True, print_to_console=True)
    Si_EF = sobol.analyze(problem, EF, calc_second_order=True, print_to_console=True)
    
    return Si_p_maxs, Si_EDV, Si_ESV, Si_stroke, Si_EF, problem["outputs"]


#####################
# Heatmap Functions #
#####################
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
        for Si in Si_s:
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

base = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/sensitivity"
LV_fname = base + "/chamber_sphere.json"
CL_fname = base + "/chamber_sphere_closed_loop.json"

svZeroDSolver_path = "/mnt/c/Users/jorda/OneDrive/Desktop/School Stuff/Yale Computational Biomechanics Research/Computational Biomechanics - svzerodsolver repo/svZeroDSolver-jt/build/svzerodsolver"


LV_baseline_input = read_file(LV_fname)
CL_baseline_input = read_file(CL_fname)


bound = [0.8,1.2] # switch from general bound (current approach) to parameter-specific bounds (see literature to discern what bounds should be)

LV_dict = create_dict(LV_baseline_input) # dictionary for (open) chamber sphere
CL_dict = create_dict(CL_baseline_input)  # dictionary for closed loop chamber sphere 


Si_y, Si_EDV, Si_ESV, Si_stroke, Si_EF, output_names = sobol_sensitivity(LV_baseline_input, LV_dict, bound, "max", "open", 6)
# Si_y, Si_EDV, Si_ESV, Si_stroke, Si_EF, output_names = sobol_sensitivity(CL_baseline_input, CL_dict, bound, "max", "closed")


# # ###############################
# # # Data Load (if errors occur) #
# # ###############################

# # # Load raw inputs, given errors or failures. 
# # raw = np.load("./raw_outputs.npy")
# # print(raw.shape)

# # p_maxs, EDV, ESV, stroke, EF = raw[0], raw[1], raw[2], raw[3], raw[4]

# # for arr in [p_maxs, EDV, ESV, stroke, EF]:
# #     nan_mask = np.isnan(arr)
# #     if nan_mask.any():
# #         arr[nan_mask] = np.nanmedian(arr)


# # param_indicators, param_map = create_indicators(LV_dict)
# # num_vars = len(param_indicators)

# # problem = {
# #     'num_vars': num_vars,
# #     'names': param_indicators,
# #     'bounds': [bound] * num_vars,
# #     'outputs': ["p_maxs", "EDV", "ESV", "stroke", "EF"]
# # }

# # # run sobol analysis on the correctly-sized arrays
# # Si_y = sobol.analyze(problem, p_maxs, calc_second_order=True, print_to_console=True)
# # Si_EDV = sobol.analyze(problem, EDV, calc_second_order=True, print_to_console=True)
# # Si_ESV = sobol.analyze(problem, ESV, calc_second_order=True, print_to_console=True)
# # Si_stroke = sobol.analyze(problem, stroke, calc_second_order=True, print_to_console=True)
# # Si_EF = sobol.analyze(problem, EF, calc_second_order=True, print_to_console=True)

# # output_names = problem["outputs"]

# # #################
# # # END Data Load #
# # #################


SIs = [Si_y, Si_EDV, Si_ESV, Si_stroke, Si_EF]

# safe_save("./Results/SIs_6_2_2026_closedloop_samplesize16_bound0.4_test1", SIs)

SA_heatmap(SIs, ["Max Pressure","EDV","ESV","STROKE","EF"], LV_dict)
# # print(f"S2 values:\n {Si_LV[S2]}")

# # checking to see if zerod solver is from a compiled native extension library or my locally built executable
# print(f"{zerod.__file__}\n")

