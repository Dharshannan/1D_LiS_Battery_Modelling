import func_potentials as func
import numpy as np

# =============================================================================
# Create a class variable for 1D LiS Potential Boundary Value Model
# =============================================================================
## Carry out variable substitution in class variable
## Simple case initiate class with variable values as before
## This class variable will be for the boundary value problem:
## to calculate the initial values for phi1 (solid state potential) and phi2 (electrolyte potential)

class LiS_Potential_Model:
        
    def __init__(self, x):
        ## For now we do not need to redefine the constant parameters (*Do this for future model)
        ## x is the variable values array ##
        self.x = x
        
    def get_func_vals(self):
        # =============================================================================
        # Reference the u array and jacobian from the func_1D_test.py script 
        # =============================================================================
        ## Reference u_list
        self.u_list = func.u_func_array
    
        ## Reference jacobian
        self.jacob = func.jacob
        
    ## Now we define a function to return the jacobian with variable values substitution done
    ## Use the code from func_1D_test.py
    # =============================================================================
    # Function to return Jacobian with substitution
    # =============================================================================
    def jacobian(self, prevt_vals, dx, h):
        ## Initiate by getting function values for jacobian
        ## prevt values not required as the interation is olny for the initial time step
        self.get_func_vals()
        x = self.x # Format similar to how the ovr_var_init list in func_1D_test.py script
        dx_dt = [dx, h]
        
        ## Logic to carry out substitution below:
        num_var = len(x) ## Number of variables (w/o spatial discretisation)
        num_space = len(x[0]) ## Number of spatially discretised points
        num_rows = len(self.jacob) ## Number of jacobian rows
        row_label = func.row_label ## Import row_label from func_1D_test.py script
        
        ovr_subs =[] ## List to hold all substitution values for each row
        for i in range(num_rows):
            ## Handle for x0 boundary point variables ##
            if row_label[i][1] == 0:
                var_subs = []
                for j in range(num_var):
                    prevt = x[j][0]
                    prevx = 0 ## There is no prevx variable at x0 boundary
                    currx = x[j][0]
                    postx = x[j][1]
                    var_subs.append([prevt, prevx, currx, postx])
            
                ovr_subs.append(var_subs)
                
            ## Handle for xL bounday point variables ##
            elif row_label[i][1] == num_space-1: ## Can be generalized when knowing the number of spatially dicretized points
                var_subs = []
                for j in range(num_var):
                    prevt = x[j][num_space-1]
                    prevx = x[j][num_space-2]
                    currx = x[j][num_space-1]
                    postx = 0 ## There is no postx variable at xL boundary
                    var_subs.append([prevt, prevx, currx, postx])
            
                ovr_subs.append(var_subs)
                
            ## Handle for all other points ##
            else:
                var_subs = []
                k = row_label[i][1]
                for j in range(num_var):
                    prevt = x[j][k]
                    prevx = x[j][k-1]
                    currx = x[j][k]
                    postx = x[j][k+1]
                    var_subs.append([prevt, prevx, currx, postx])
                ovr_subs.append(var_subs)
                
        ## Unpack substitution list and add dx & dt(h)
        for i in range(len(ovr_subs)):
            ovr_subs[i] = [item for sublist in ovr_subs[i] for item in sublist] + dx_dt
        
        ## Now we carry out substitution
        jacob_res = np.zeros((len(self.jacob), len(self.jacob[0])))
        for i in range(len(self.jacob)):
            for j in range(len(self.jacob[i])):
                if self.jacob[i][j] != 0:
                    ## The Jacobian is always a 2D square Matrix
                    jacob_res[i][j] = float(self.jacob[i][j](*ovr_subs[i]))
                
        # Return jacob array
        jacob_res = np.asarray(jacob_res, dtype='float64')
        return(jacob_res)
    
    ## Now we define a function to return the u array with variable values substitution done
    ## Use the code from previous jacobian function
    # =============================================================================
    # Function to return u/function array with substitution
    # =============================================================================
    def f(self, prevt_vals, dx, h):
        ## Initiate by getting function values for u_array
        ## prevt values not required as the interation is olny for the initial time step
        self.get_func_vals()
        x = self.x # Format similar to how the ovr_var_init list in func_1D_test.py script
        dx_dt = [dx, h]
        
        ## Logic to carry out substitution below:
        num_var = len(x) ## Number of variables (w/o spatial discretisation)
        num_space = len(x[0]) ## Number of spatially discretised points
        num_rows = len(self.jacob) ## Number of jacobian rows
        row_label = func.row_label ## Import row_label from func_1D_test.py script
        
        ovr_subs =[] ## List to hold all substitution values for each row
        for i in range(num_rows):
            ## Handle for x0 boundary point variables ##
            if row_label[i][1] == 0:
                var_subs = []
                for j in range(num_var):
                    prevt = x[j][0]
                    prevx = 0 ## There is no prevx variable at x0 boundary
                    currx = x[j][0]
                    postx = x[j][1]
                    var_subs.append([prevt, prevx, currx, postx])
            
                ovr_subs.append(var_subs)
                
            ## Handle for xL bounday point variables ##
            elif row_label[i][1] == num_space-1: ## Can be generalized when knowing the number of spatially dicretized points
                var_subs = []
                for j in range(num_var):
                    prevt = x[j][num_space-1]
                    prevx = x[j][num_space-2]
                    currx = x[j][num_space-1]
                    postx = 0 ## There is no postx variable at xL boundary
                    var_subs.append([prevt, prevx, currx, postx])
            
                ovr_subs.append(var_subs)
                
            ## Handle for all other points ##
            else:
                var_subs = []
                k = row_label[i][1]
                for j in range(num_var):
                    prevt = x[j][k]
                    prevx = x[j][k-1]
                    currx = x[j][k]
                    postx = x[j][k+1]
                    var_subs.append([prevt, prevx, currx, postx])
                ovr_subs.append(var_subs)
                
        ## Unpack substitution list and add dx & dt(h)
        for i in range(len(ovr_subs)):
            ovr_subs[i] = [item for sublist in ovr_subs[i] for item in sublist] + dx_dt
            
        ## Now we carry out substitution
        u_array = np.zeros(len(self.u_list))
        for i in range(len(self.u_list)):
            u_array[i] = float(self.u_list[i](*ovr_subs[i]))
            
        ## Return u array
        u_array = np.asarray(u_array, dtype='float64')
        return(u_array)
    
    ## Define function to update parameter/constant values versatilely 
    def update_parameters(self, **kwargs):
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)
    
# =============================================================================
# End of class variable
# =============================================================================

# =============================================================================
# =============================================================================
# =============================================================================
#         Now we define the solver function for time iterations
# =============================================================================
# =============================================================================
# =============================================================================
