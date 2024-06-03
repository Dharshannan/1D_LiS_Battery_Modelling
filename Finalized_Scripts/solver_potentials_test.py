import numpy as np
from LiS_Potential_Model import LiS_Potential_Model as LiSModel
import func_potentials as func
import warnings
import matplotlib.pyplot as plt
from numba import njit, jit
import timeit
import os

## This script is to test the solver before fully implementing it as a function ##
## The function will be implemented into the LiS_Potential_Model.py script ##

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

## Define variable (Potentials) initial guess values
phi1_init = 0
phi2_init = 0
init_list = [phi1_init, phi2_init]

## Form variable list in format required to be passed into solver function and model class
x_var = []
for i in range(len(init_list)):
    x_var.append([])
    for j in range(num_space):
        x_var[i].append(init_list[i])
        
x_var[0][0] = 0 ## Set phi1 value at anode to be zero

## We apply to the solver for testing
def LiS_Solver(x_var, # This x_var variable will be a list containing all the variable values  
               t_end, h0, break_voltage, state = None, t0 = 0, backtracked = None, 
               params_backtrack = {}, upd_params = {}):
    
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

    b = 0
    breakpoint1 = 0
    min_h = 1e-4 # Minimum step size
    max_h = 1.25 # Maximum step size
    max_jacobian = 0 # Variable to store maximum jacobian determinant
    
    # Set the warning filter to raise RuntimeWarning as an exception
    warnings.simplefilter("error", category=RuntimeWarning)
    
    ## Now start time iteration loop
    i = 1
    #start = timeit.default_timer() ## Start timer (Can be used in other scripts instead when calling the solver function)
    while t[-1] < t_end:
        
        try:
            # Initialize with guess values for each variable (use previous values as initial guesses)
            x_varguess = x_var[:, i-1]
            h = h_step[i-1]
            t_current = t[i-1]
    
            # Define the damping to be 1 at the start, we will dynamically update this if necessary
            lamda = 1 # Damping Factor
            damping_update_factor = 0.25
            damping_min = 1e-8
            regularization_factor = 5e-4
            
            while True:
                # Now calculate u (function) values and define jacobian elements using model class
                
                # Create a list to store the old guess values
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
                #print(u_array)
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
                    max_ratio = abs(jacob_array[i-2])/max_jacobian
                    ## These values need configuration
                    if max_ratio >= 1.2: ## This indicates step needs to be reduced and parameter backtracking
                        ## Define recursive function call for parameter backtracking
                        ## Recursive call to only solve for 1 iteration:
                        t02 = 0
                        t_end2 = t02 + h
                        new_guess = LiS_Solver(x_varguess, 
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
                        
                        # =============================================================================
                        # Define Line Search Method to further optimize the step size           
                        # =============================================================================
                        delta2 = -numpy_solve_numba(jacob, u_array)
                        alpha = 0.3
                        beta = 0.05
                        var_val = x.copy()
                        unew_array = u_array.copy()
        
                        while np.linalg.norm(unew_array) > np.linalg.norm(u_array + alpha*h*np.dot(jacob, delta2)):
                            upd_x = var_val.reshape((len(jacob),)) + h*delta2
                            upd_x = (upd_x).reshape((len(x), num_space))
                            model3 = LiSModel(upd_x) ## Initialize the model
                            ## Change any values if a new value wants to be used apart from the default ones in the model class
                            model3.update_parameters(**upd_params)
                            unew_array = model3.f(x_var[:, -1], dx, h)
                            h *= beta
                            if h < min_h:
                                break
                            #print(h, I*t[i-1]/3600)
                        
                        # Further Update h (step-size)
                        h_new = max(h*(0.25), min_h) ## Saturate at minimum step size
                        
                    elif max_ratio <= 1.0: ## This indicates step size can be increased 
                        h_new = min(h/(0.75), max_h) ## Saturate at maximum step size
                        
                    else:
                        h_new = h
                        
                else:
                    h_new = h
                
                ## Now we solve as usual the new step size will be implemented in the next iteration
                jacob = jacob + regularization_factor * np.eye(len(jacob))
                # Calculate new values via Newton-Raphson
                delta = -numpy_solve_numba(jacob, u_array)
                new_val = x.reshape((len(jacob),)) + lamda*delta
                new_val = (new_val).reshape((len(x), num_space))
                #print(new_val)
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
                    lamda /= (damping_update_factor**n_damp)
                
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
                if np.all(err_list<tol): #and np.all(abs(unew_array)<1e5):
                    # If the error is small values have converged hence, update the new variables and break the while loop
                    t_next = t_current + h_new
                    t.append(t_next)
                    h_step.append(h_new)
                    jacob_array.append(norm_jacob)
                    print('================================================')
                    print('Maximum function value: ', max(abs(unew_array)))
                        
                    ## Concatenate the new values to the exsisting list
                    x_var = np.concatenate((x_var, new_val[:, np.newaxis]), axis=1)
                    break
    
                else:
                    # Update guess values to be new values
                    x_varguess = new_val 
                    #print(new_val)
    
            b = b + 1
            i = i + 1
            
            print(f"No of iterations: {b}/{int(t_end/h)}")
            print('================================================')
            
            if (t[-1] >= t_end):
    # =============================================================================
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var
                t_array = np.asarray(t)
            
                return(var_array)
                break
            
        ## When a warning is encountered break the loop: ##
        except RuntimeWarning as warning:
            print("Runtime Warning Encountered at Capacity:", t[i-1]/3600)
            warnings.warn(warning)  # Raise the RuntimeWarning again
            break
        
## Test ##
index_remove = func.index_separator_points
start = timeit.default_timer() ## Start timer
solution = LiS_Solver(x_var, 500, 0.5, 2) ## Can be as low as 200 itrs
potentials = solution[:,-1]
# =============================================================================
# print(potentials[:,-1])
# print('Number of spatial points: ', len(potentials[:,1][0]))
# print("The time taken for solver completion :", (timeit.default_timer() - start), "s")
# phi1_vals = np.delete(potentials[:,-1][0], index_remove)
# x_vals_phi1 = np.delete(func.x_space, index_remove)
# ## Plotting ##
# plt.plot(x_vals_phi1[1:], phi1_vals[1:], label=f'phi1 (initial guess = {phi1_init})', marker='^')
# plt.plot(func.x_space, potentials[:,-1][1], label=f'phi2 (initial guess = {phi2_init})')
# plt.plot(func.x_space, potentials[:,-1][0] - potentials[:,-1][1], label='phi1 - phi2', linestyle='dashed')
# plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
# plt.xlabel('Cell Distance (m)')
# plt.ylabel('Potential (V)')
# plt.title(f'Potential vs Cell Distance for {len(potentials[:,1][0])} points')
# path, filename = os.path.split(os.path.realpath(__file__))
# #plt.savefig(f'{path}\Potentials Distribution {len(potentials[:,1][0])} points.png', dpi=1500, bbox_inches='tight')
# #plt.tight_layout()
# #plt.subplots_adjust(right=0.8)
# plt.show()
# =============================================================================
