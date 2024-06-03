import numpy as np
from LiS_Model import LiS_1D_Model as LiSModel
import func_1D_LiS as func
import warnings
import matplotlib.pyplot as plt
from numba import njit, jit
import timeit
import os
import solver_potentials_test as potentials
import time
import sys
import generate_square_wave

## This script is to test the solver before fully implementing it as a function ##
## The function will be implemented into the LiS_Model.py script ##

## Function to safely calculate the determinant in-case of overflow errors
def safe_det(matrix, max_value=1e300):
    matrix = matrix.copy() ## Copy to avoid modifying original matrix
    ## Remove penalty factor from jacobian before calculating determinant
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i == j and i in func.index_to_delete:
                matrix[i][j] = 1 ## Remove penalty factored elements
    try:
        with np.errstate(all='raise'):
            determinant = min(np.abs(np.linalg.det(matrix)), max_value)

    except (np.linalg.LinAlgError, FloatingPointError, RuntimeWarning):
        # Handle LinAlgError, FloatingPointError, or RuntimeWarning by setting to a default value
        determinant = max_value

    return determinant

## Function for the Gustaffson adaptive ste size approach
## k1 & k2 values that tested to be good (add to below):
## (k1=2, k2=0.001), 
def upd_step(R, h, i):
    k = 1
    k1 = 2
    k2 = 0.001
    tol = 1e-5
    denom = R[i-1] + 1e-10
    
    if i == 1:
        h_new = ((tol/denom)**(1/k))*(h[i-1])
        
    else:
        h_new = (h[i-1]/h[i-2])*((tol/denom)**(k2/k))*((R[i-2]/denom)**(k1/k))*h[i-1]
    
    return(h_new)

## Function for faster matrix solutions with numba for numpy (just in time compilation)
@njit(parallel=True)
def numpy_solve_numba(A, b):
    return np.linalg.solve(A, b)

## Import number of spatially discretised points
num_space = func.num_space
dx = func.delta_x

## Define variable initial guess values
C_Li = 1001
C_s8 = 19 #(If running without precipitation/dissolution, multiply C_s8 by a large value like 100 to account for capacity issues)
C_s8_2 = 0.18
C_s6 = 0.32
C_s4 = 0.02
C_s2 = 5.23e-7
C_s1 = 8.27e-10
C_An = 1000
e_cath = 0.7
e_sep = 0.5
e_s8_cath = 0.166
e_s8_sep = 1e-12
e_Li2s_cath = 1e-7
e_Li2s_sep = 1e-7

init_list = [C_Li, C_s8, C_s8_2, C_s6, C_s4, C_s2, C_s1, C_An, e_cath, e_sep, e_s8_cath, e_s8_sep, e_Li2s_cath, e_Li2s_sep]


## Form variable list in format required to be passed into solver function and model class
x_var = []
for i in range(len(init_list)):
    x_var.append([])
    for j in range(num_space):
        x_var[i].append(init_list[i])

print('\n')
print('Solved Potentials Initial Condition...')
potential_init = potentials.potentials
x_var = x_var + [list(potential_init[0])] + [list(potential_init[1])]
time.sleep(2)

## We apply to the solver for testing
def LiS_Solver(x_var, I, # This x_var variable will be a list containing all the variable values  
               t_end, h0, break_voltage, state = None, t0 = 0, backtracked = None, conv_critereon = 'var_tol', 
               params_backtrack = {}, upd_params = {}, max_step=1.5, min_step=0.5, job_idx=0, filename='variables.npz', adaptive_step='pure_res'):
    
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
            
            # =============================================================================
            ## Get pulse current values (Modify Accordingly)
            # =============================================================================
            ## For constant:
            I_app = I
            ## For square wave:
            start_time = t0
            end_time = t_end
            high_value = I
            low_value = 0
            high_pulse_length = (end_time/75)*0.7
            low_pulse_length = (end_time/75)*0.3
            #I_app = generate_square_wave.ret_val_square(t_current, start_time, end_time, high_value, low_value, high_pulse_length, low_pulse_length)
            ## For sinusoidal wave
            freq = 2*np.pi*0.00025 ## 2.5e-4 Hz
            #I_app = generate_square_wave.ret_val_sin(t_current, I, freq)
            ## For discharge-rest-charge
            high_time = 21000
            low_time = 34000
            charge_curr = -0.4
            #I_app = generate_square_wave.ret_val_steps(t_current, high_value, charge_curr, high_time, low_time)
            
            # =============================================================================
            # Solver Hyper-Parameters          
            # =============================================================================
            # Define the damping to be 1 at the start, we will dynamically update this if necessary
            lamda = 1 # Damping Factor
            ## Damping update of 1 (no adaptive damping) seems to work best
            damping_update_factor = 1 #0.95 #0.5 #0.25
            damping_min = 1e-5
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
                model = LiSModel(x, I_app) ## Initialize the model
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
                #     ### This is a placeholder for backtracking implementation ###
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
                            model = LiSModel(x_upd, I_app)
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
                model2 = LiSModel(new_val, I_app) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model2.update_parameters(**upd_params)
                unew_array = model2.f(x_var[:, -1], dx, h)
    
                # Compute the ratio of actual reduction to predicted reduction
                actual_reduction = np.linalg.norm(u_array) - np.linalg.norm(unew_array)
                predicted_reduction = np.linalg.norm(u_array) - np.linalg.norm(u_array + jacob @ delta)
                ratio = abs(actual_reduction / (predicted_reduction + 1e-10))
    
                # Update the damping factor based on the ratio (hyper-parameters may require tuning)
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
                if conv_critereon == 'var_tol':
                    critereon = np.all(err_list<tol)
                
                elif conv_critereon == 'var_res_tol':
                    critereon = np.all(err_list<tol) and np.all(abs(unew_array)<tol)
                
                elif conv_critereon == 'res_tol':
                    critereon = np.all(abs(unew_array)<tol)
                    
                if critereon or itrs > max_iterations: #and np.all(abs(unew_array)<tol):
                    # =============================================================================
                    # Residual difference adaptive step size approach                
                    # =============================================================================
                    R.append(max(abs(unew_array))) ## Append maximum residual
                    ## Adaptive step size method (hyper-parameters may require tuning)
                    if i > 100: ## Run 1st 100 iterations with small step_size to stabilize residuals
                        if adaptive_step == 'pure_res': ## If pure residual adaptive step is used
                            temp_res1 = (R[i-1] - R[i-2])
                            res_grads.append(temp_res1)
                            max_res = max(temp_res1, max_res)
                            temp_res2 = temp_res1/(max_res + 1e-10)
                            res_grads_ratio.append(temp_res2)
                            if temp_res2 >= 1e-2:
                                h_new = max(h*(0.25), min_h)
                            else:
                                h_new = min(h/(0.75), max_h)
                                
                        elif adaptive_step == 'gustaffson': ## If Gustaffson adaptive step is used
                            h_new = upd_step(R, h_step, i)
                            h_new = max(min(h_new, max_h), min_h)
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
                print("Discharge Capacity: ", abs(I)*t[i-1]/3600, 'Ah')
                print(f"Time elapsed: {(timeit.default_timer() - start)/3600}h")
                #print("Jacobian Determinant: ", norm_jacob)
                print('================================================')
                sys.stdout.flush()  # Flush the output to ensure it's immediately visible
            phi1 = x_var[-2]
            phi2 = x_var[-1]
            phi1_cath = phi1[:, -1]
            phi2_anod = phi2[:, 0]
            voltage = phi1_cath - phi2_anod
            #Ah = func.I_app*t[i-1]/3600
            
            if state == 'Discharge' and ((t[-1] >= t_end) or voltage[-1] < break_voltage): ## For discharge
                var_array = x_var
                t_array = np.asarray(t)
                residuals = np.asarray(residuals)
                np.savez(filename, solved=var_array, time=t_array, runtime=timeit.default_timer() - start, residuals=residuals)
                return(var_array, t_array, timeit.default_timer() - start, residuals)
                break
            
            elif state == 'Charge' and ((t[-1] >= t_end) or voltage[-1] > break_voltage): ## For charge
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

def parallel_sim(h_array, x_var):
    ## Function takes in the array of step sizes to test and runs parallel simulations
    ## Maximum number of operations
    max_operations = os.cpu_count()
    n_operations = len(h_array)
    I_applied = 0.5
    
    ## Call joblib for parallel runs
    joblib.Parallel(n_jobs=min(n_operations,max_operations), prefer="threads")(
        joblib.delayed(LiS_Solver)(x_var, I_applied, 250000, 5e-1, 2.0, max_step=h, min_step=5, job_idx=idx, filename='var_array_test.npz', conv_critereon='var_res_tol', adaptive_step='pure_res', state='Discharge') for idx, h in enumerate(h_array))

## Run parallel simulations by defining the step size to test array
h_array = [10] ## Maximum step sizes to test

# call parallel function to execute parallel simulations
parallel_sim(h_array, x_var)