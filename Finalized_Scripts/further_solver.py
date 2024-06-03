import numpy as np
from LiS_Model import LiS_1D_Model as LiSModel
import func_1D_LiS as func
import warnings
import matplotlib.pyplot as plt
from numba import njit, jit
import timeit
import os
import sys

## This script is to test the solver before fully implementing it as a function ##
## The function will be implemented into the LiS_Model.py script ##

## Function to safely calculate the determinant in-case of overflow errors
def safe_det(matrix, max_value=1e300):
    try:
        with np.errstate(all='raise'):
            determinant = min(np.abs(np.linalg.det(matrix)), max_value)

    except (np.linalg.LinAlgError, FloatingPointError, RuntimeWarning):
        # Handle LinAlgError, FloatingPointError, or RuntimeWarning by setting to a default value
        determinant = max_value

    return determinant

## Function for faster matrix solutions with numba for numpy (just in time compilation)
@njit(parallel=True)
def numpy_solve_numba(A, b):
    return np.linalg.solve(A, b)

## Import number of spatially discretised points
num_space = func.num_space
dx = func.delta_x

## Take the end point of the previous solution and pass into function for further iterations
data = np.load('var_array_0.5A_constant.npz', allow_pickle=True)
overall_array = data['solved']
time = data['time']
species = overall_array
charge_start = -1 ## Index to start charge
species_endpoint = species[:, charge_start] ## Take the final values of all variables
time_end = time[charge_start]

## We apply to the solver for testing
def LiS_Solver(x_var, # This x_var variable will be a list containing all the variable values  
               t_end, h0, break_voltage, state = None, t0 = 0, backtracked = None, 
               params_backtrack = {}, upd_params = {}, max_step=1.5, min_step=0.5, job_idx=0, filename='variables.npz'):
    
    ## Define state argument to handle breakpoints for charge and discharge seperately
    if state == None:
        state = 'Discharge' # Default to Discharge
    
    if state != None and state != 'Discharge' and state != 'Charge':
        raise ValueError('state can only be None, Discharge or Charge')
        
    ## The i above stands for initial, so s8i stands for initial s8 value ##
    t0 = t0
    h0 = h0
    t_end = t_end

    ## Initialize the variable values and arrays
    x_var = np.asarray(x_var) # convert to numpy array
    x_var = x_var[:, np.newaxis] # Append values to new axis to create 2D array
    t = [t0]
    h_step = [h0]
    jacob_array = []
    R = [] ## Array to hold the maximum residuals that occur at each iteration

    b = 0
    breakpoint1 = 0
    min_h = min_step # Minimum step size
    max_h = max_step # Maximum step size
    max_jacobian = 0 # Variable to store maximum jacobian determinant
    err_upd_h = [] ## empty array initialization
    max_res = 0 ## Variable to store maximum residual difference
    res_grads = [] ## Array to store residual differences
    res_grads_ratio = [] ## Array to store residual differences ratio
     
    # Set the warning filter to raise RuntimeWarning as an exception
    warnings.simplefilter("error", category=RuntimeWarning)
    
    ## Now start time iteration loop
    i = 1
    start = timeit.default_timer() ## Start timer
    residuals = [] ## Initialize empty array to hold residual arrays at each time step
    while t[-1] < t_end:
        
        try:
            # Initialize with guess values for each variable (use previous values as initial guesses)
            x_varguess = x_var[:, i-1]
            h = h_step[i-1]
            t_current = t[i-1]
    
            # Define the damping to be 1 at the start, we will dynamically update this if necessary
            lamda = 1 # Damping Factor
            damping_update_factor = 1 #0.25
            damping_min = 1e-8
            ## NOTE: The regularization factor is quite sensitive here ##
            regularization_factor = 1e-8 #5e-7 #5e-4
            max_iterations = 50
            itrs = 0
            
            while True:
                # Now calculate u (function) values and define jacobian elements using model class
                
                # Create a list to store the old guess values
                itrs += 1
                x = x_varguess
                
                ## Call model class and get the function arrays and jacobian
                model = LiSModel(x) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model.update_parameters(**upd_params)
                ## Change parameter values if recursion has happenend (Backtrack to set value)
                if backtracked == True:
                    model.update_parameters(**params_backtrack)
                ## Now solve as usual with new backtracked parameter value
                u_array = model.f(x_var[:, -1], dx, h)
                jacob = model.jacobian(x_var[:, -1], dx, h)
                
                ### Now we calculate the determinant of the Jacobian and check to alter step size ###
                norm_jacob = safe_det(jacob)
                max_ratio = 0
    
                # =============================================================================
                #     ### This will dynamically update the step size every iteration ###
                # =============================================================================
                if i > 1: ## Only check after 1st iteration ##
                    ## Continuously update the maximum value of determinant of Jacobian encountered
                    max_jacobian = max(max_jacobian, abs(jacob_array[i-3]))
                    ## Calculate the ratio of current determinant to maximum for Jacobian
                    max_ratio = abs(jacob_array[i-2])/(max_jacobian + 1e-8)
                    ## These values need configuration
                    if max_ratio>1e-200 and itrs>2: ## or max(err_upd_h)>h_tol: ## This indicates step needs to be reduced and parameter backtracking
                        ## Define recursive function call for parameter backtracking
                        ## Recursive call to only solve for 1 iteration:
                        ## Only call backtracking if parameters provided
                        if params_backtrack != {}:
                            t02 = 0
                            t_end2 = t02 + h
                            new_guess, unused, unused2, unused3 = LiS_Solver(x_varguess, 
                                                   t_end2, h, break_voltage, state=state, t0=t02, backtracked=True, 
                                                   params_backtrack=params_backtrack, upd_params=upd_params)
                            
                            ## Now update the guess values and run solver by updating u_array and jacobian
                            x_upd = np.asarray(new_guess[:, -1])
                            # Update the model
                            model = LiSModel(x_upd)
                            ## Change any values if a new value wants to be used apart from the default ones in the model class
                            model.update_parameters(**upd_params)
                            u_array = model.f(x_var[:, -1], dx, h)
                            jacob = model.jacobian(x_var[:, -1], dx, h)
                            x = x_upd ## Use updated guess values from backtracking
                        
                    else:
                        pass
                        
                else:
                    pass
                
                ## Now we solve as usual the new step size will be implemented in the next iteration
                jacob = jacob + regularization_factor * np.eye(len(jacob))
                # Calculate new values via Newton-Raphson
                delta = -numpy_solve_numba(jacob, u_array)
                new_val = x.reshape((len(jacob),)) + lamda*delta
                new_val = (new_val).reshape((len(x), num_space))
                for v in range(len(new_val) - 2): ## Test with all variables abs (including phi1 & phi2) (*UPDATE DOES NOT WORK LOL :))
                    new_val[v] = np.abs(new_val[v]) ## Absolute the concentrations but not potentials

                # =============================================================================
                # ## Next up code loop & error handling ##
                # =============================================================================
                model2 = LiSModel(new_val) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model2.update_parameters(**upd_params)
                unew_array = model2.f(x_var[:, -1], dx, h)
    
                # Compute the ratio of actual reduction to predicted reduction
                actual_reduction = np.linalg.norm(u_array) - np.linalg.norm(unew_array)
                predicted_reduction = np.linalg.norm(u_array) - np.linalg.norm(u_array + jacob @ delta)
                ratio = abs(actual_reduction / (predicted_reduction + 1e-10))
    
                # Update the damping factor based on the ratio
                n_damp = 1
                if ratio > 1e-3:
                    lamda *= (damping_update_factor**n_damp)
                elif ratio < 1e-4:
                    increase_factor = 1
                    lamda /= (increase_factor*damping_update_factor**n_damp)
                
                # Ensure the damping factor does not go below the minimum value
                lamda = max(lamda, damping_min)
    
                # Calculate error between new_val and old_guess
                x_to_check = x.reshape((len(jacob),))
                new_val_to_check = new_val.reshape((len(jacob),))
                err_list = np.zeros((len(x_to_check),))
                epsilon = 1e-10
                for j in range(len(err_list)):
                    err_list[j] = abs((new_val_to_check[j] - x_to_check[j])/(x_to_check[j] + epsilon))
                
                # Check if all absolute difference/err is smaller than a specified tolerance
                tol = 1e-2
                if (np.all(err_list<tol)) or itrs > max_iterations: #and np.all(abs(unew_array)<tol):
                    # =============================================================================
                    # Residual difference adaptive step size approach                
                    # =============================================================================
                    R.append(max(abs(unew_array))) ## Append maximum residual
                    ## Adaptive step size method
                    if i > 100:
                        temp_res1 = (R[i-1] - R[i-2])
                        res_grads.append(temp_res1)
                        max_res = max(temp_res1, max_res)
                        temp_res2 = temp_res1/(max_res + 1e-10)
                        res_grads_ratio.append(temp_res2)
                        if temp_res2 >= 1e-2:
                            h_new = max(h*(0.25), min_h)
                        else:
                            h_new = min(h/(0.75), max_h)
                    else:
                        h_new = h
                    # If the error is small values have converged hence, update the new variables and break the while loop
                    t_next = t_current + h_new
                    t.append(t_next)
                    h_step.append(h_new)
                    jacob_array.append(norm_jacob)
                    index_max = np.argmax(unew_array)
                    residuals.append(abs(unew_array)) ## Append the residual arrays
                    
                    if job_idx == 0:
                        print('================================================')
                        print('Maximum function value: ', max(abs(unew_array)))
                        print('Maximum function occurs at: ', func.row_label[index_max][0], func.row_label[index_max][1])
                        print('Number of iterations taken for convergence: ', itrs)
                        print('Voltage at convergence: ', new_val[-2, -1] - new_val[-1, 0])
                        
                    ## Concatenate the new values to the exsisting list
                    x_var = np.concatenate((x_var, new_val[:, np.newaxis]), axis=1)
                    break
    
                else:
                    # Update guess values to be new values
                    x_varguess = new_val 
    
            b = b + 1
            i = i + 1
            
            if job_idx == 0:
                print(f"No of iterations: {b}/{int((t_end-t0)/h)}")
                print("Discharge Capacity: ", abs(func.I_app)*t[i-1]/3600, 'Ah')
                print(f"Time elapsed: {(timeit.default_timer() - start)/3600}h")
                print('================================================')
                sys.stdout.flush()  # Flush the output to ensure it's immediately visible
            phi1 = x_var[-2]
            phi2 = x_var[-1]
            phi1_cath = phi1[:, -1]
            phi2_anod = phi2[:, 0]
            voltage = phi1_cath - phi2_anod
            Ah = abs(func.I_app)*t[i-1]/3600
            
            if (t[-1] >= t_end) or voltage[-1] > break_voltage or Ah>=10: ## Stop at 10Ah discharge or break after exceed charge voltage
                var_array = x_var
                t_array = np.asarray(t)
                residuals = np.asarray(residuals)
                np.savez(filename, solved=var_array, time=t_array, runtime=timeit.default_timer() - start, residuals=residuals)
                return(var_array, t_array, timeit.default_timer() - start, residuals)
                break
            
        ## When a warning is encountered break the loop: ##
        except RuntimeWarning as warning:
            if job_idx == 0:
                print("Runtime Warning Encountered at time (h):", t[i-1]/3600)
                print('Voltage at error: ', voltage[-1])
                sys.stdout.flush()  # Flush the output to ensure it's immediately visible
            t = np.asarray(t)
            residuals = np.asarray(residuals)
            np.savez(filename, solved=x_var, time=t, runtime=timeit.default_timer() - start, residuals=residuals)
            return(x_var, t, timeit.default_timer() - start, residuals)
            warnings.warn(warning)  # Raise the RuntimeWarning again
            break
        
        ## Keyboard interrupt to stop simulation and return results
        except KeyboardInterrupt:
            if job_idx == 0:
                print('Simulation Run Stopped: Returning Data Collected...')
                sys.stdout.flush()  # Flush the output to ensure it's immediately visible
            t = np.asarray(t)
            residuals = np.asarray(residuals)
            np.savez(filename, solved=x_var, time=t, runtime=timeit.default_timer() - start, residuals=residuals)
            return(x_var, t, timeit.default_timer() - start, residuals)
            break

## Test ##
# =============================================================================
# Define a function to parallelize the simulations
# =============================================================================
## Import joblib to parallelize the simulations
import joblib

def parallel_sim(h_array, x_var, tend):
    ## Function takes in the array of step sizes to test and runs parallel simulations
    ## Maximum number of operations
    max_operations = os.cpu_count()
    n_operations = len(h_array)
    
    ## Call joblib for parallel runs
    joblib.Parallel(n_jobs=min(n_operations,max_operations), prefer="threads")(
        joblib.delayed(LiS_Solver)(x_var, 250000, 5e-1, 2.5, t0=tend, max_step=h, min_step=10, job_idx=idx, filename=f'var_array_charge(2).npz') for idx, h in enumerate(h_array))

## Run parallel simulations by defining the step size to test array
h_array = [10] ## Maximum step sizes to test

# call parallel function to execute parallel simulations
parallel_sim(h_array, species_endpoint, time_end)