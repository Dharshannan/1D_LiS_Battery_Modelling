from sympy import symbols, diff, exp, sqrt, log, lambdify
import numpy as np
import timeit

## Loading bar function:
def update_loading_bar(iteration, total):
    percent_complete = (iteration / total) * 100
    loading_bar = f"[{'#' * int(percent_complete / 2)}{' ' * (50 - int(percent_complete / 2))}] {percent_complete:.2f}%"
    print('\033[K' + loading_bar, end='\r')  # \033[K clears the line

# =============================================================================
# Here we define the boundary value problem to solve for the phi1 & phi2 
# potential initial conditions to be used within the time stepping solver 
# Implementation of Parabolic formulations for potentials initial conditions
# Turn elliptic equations to time dependant parabolic equations
# Test if system equilibrates for the parabolic equations
# =============================================================================

## This model is the full 6-stage seperator + cathode model (**Added Anion Salt, A- species) ##
## Simplified only by assuming constant porosity and no precipitation dynamics ##

## Define all parameters as symbols

F = 96485.3321233100184
R = 8.3145
T = 292.15
L_cath = 20e-6
L_sep = 25e-6
A = 0.28
av = 132762
e_cath = 0.7 # Porosity coefficient at cathode
e_sep = 0.5 # Porosity coefficient at seperator
sigma = 1
i01 = 0.5
i02 = 1.9
i03 = 0.02
i04 = 0.02
i05 = 2e-4
i06 = 2e-9
U1 = 0
U2 = 2.41 #2.35 #2.41
U3 = 2.35 #2.06 #2.35
U4 = 2.23 #2.05 #2.23
U5 = 2.03
U6 = 2.01
z_Li = 1
z_s8 = 0
z_s8_2 = -2
z_s6 = -2
z_s4 = -2
z_s2 = -2
z_s = -2
z_A = -1
## Dynamic coefficients at cathode:
D_Li_cath = 0.88e-12*(e_cath**1.5)
D_s8_cath = 0.88e-11*(e_cath**1.5)
D_s8_2_cath = 3.5e-12*(e_cath**1.5)
D_s6_cath = 3.5e-12*(e_cath**1.5)
D_s4_cath = 1.75e-12*(e_cath**1.5)
D_s2_cath = 0.88e-12*(e_cath**1.5)
D_s_cath = 0.88e-12*(e_cath**1.5)
D_A_cath = 3.5e-12*(e_cath**1.5)
## Dynamic coefficients at seperator:
D_Li_sep = 0.88e-12*(e_sep**1.5)
D_s8_sep = 0.88e-11*(e_sep**1.5)
D_s8_2_sep = 3.5e-12*(e_sep**1.5)
D_s6_sep = 3.5e-12*(e_sep**1.5)
D_s4_sep = 1.75e-12*(e_sep**1.5)
D_s2_sep = 0.88e-12*(e_sep**1.5)
D_s_sep = 0.88e-12*(e_sep**1.5)
D_A_sep = 3.5e-12*(e_sep**1.5)
## These reference values are the initial conditions for the species ##
C_Li = 1001
C_s8 = 19
C_s8_2 = 0.18
C_s6 = 0.32
C_s4 = 0.02
C_s2 = 5.23e-7
C_s = 8.27e-10
C_A = 1000
I_app = 0

## Space discretisation
no_points = 14 ## 28 Points here is 50 discretized points in space
## NOTE: Above no_points is only for seperator region, points for cathode is calculated based of dx in seperator
x_space_sep = np.linspace(0, L_sep, no_points)
delta_x = x_space_sep[1] - x_space_sep[0]
x_space_cath = np.arange(L_sep, (L_sep + L_cath + delta_x), delta_x)
x_space = np.append(x_space_sep[:-1], x_space_cath)
index_sep_cath_boundary = np.where(x_space == x_space_sep[-1])[0][0]

## space and time steps
dx = symbols('dx')
dt = symbols('dt')

## Variable symbols
## 1) Current in time and space ##
phi1 = symbols('phi1') # Solid state potential
phi2 = symbols('phi2') # Electrolyte potential

## 2) Current in time before in space
phi1_prevx = symbols('phi1_prevx') # Solid state potential
phi2_prevx = symbols('phi2_prevx') # Electrolyte potential

## 3) Current in time after in space
phi1_postx = symbols('phi1_postx') # Solid state potential
phi2_postx = symbols('phi2_postx') # Electrolyte potential

## 4) Previous in time current in space
phi1_prevt = symbols('phi1_prevt') # Solid state potential
phi2_prevt = symbols('phi2_prevt') # Electrolyte potential

## Introduce lambdify symbols
phi1_syms = [phi1_prevt, phi1_prevx, phi1, phi1_postx]
phi2_syms = [phi2_prevt, phi2_prevx, phi2, phi2_postx]

phi1_syms_to_solve = [phi1_prevx, phi1, phi1_postx]
phi2_syms_to_solve = [phi2_prevx, phi2, phi2_postx]

syms = phi1_syms + phi2_syms + [dx, dt] ## For lambdification
syms_to_solve = phi1_syms_to_solve + phi2_syms_to_solve ## For Jacobian generalization

## Define dependant equations ##
U1_ref = U1 - ((R*T)/(F))*(-1*log(C_Li/1e3))
U2_ref = U2 - ((R*T)/(2*F))*log(C_s8_2/C_s8)
U3_ref = U3 - ((R*T)/(F))*(-1.5*(log(C_s8_2/1e3)) + 2*(log(C_s6/1e3)))
U4_ref = U4 - ((R*T)/(F))*(-1*log(C_s6/1e3) + 1.5*log(C_s4/1e3))
U5_ref = U5 - ((R*T)/(F))*(-0.5*log(C_s4/1e3) + log(C_s2/1e3))
U6_ref = U6 - ((R*T)/(F))*(-0.5*log(C_s2/1e3) + log(C_s/1e3))
eta1 = phi1 - phi2 - U1_ref
eta2 = phi1 - phi2 - U2_ref
eta3 = phi1 - phi2 - U3_ref
eta4 = phi1 - phi2 - U4_ref
eta5 = phi1 - phi2 - U5_ref
eta6 = phi1 - phi2 - U6_ref

i1 = i01*(exp(0.5*(F/(R*T))*eta1) - exp(-0.5*(F/(R*T))*eta1)) ## Use this
#i1 = i01*(-1*exp(-0.5*(F/(R*T))*eta1))
i2 = i02*(exp(0.5*(F/(R*T))*eta2) - exp(-0.5*(F/(R*T))*eta2))
i3 = i03*(exp(0.5*(F/(R*T))*eta3) - exp(-0.5*(F/(R*T))*eta3))
i4 = i04*(exp(0.5*(F/(R*T))*eta4) - exp(-0.5*(F/(R*T))*eta4))
i5 = i05*(exp(0.5*(F/(R*T))*eta5) - exp(-0.5*(F/(R*T))*eta5))
i6 = i06*(exp(0.5*(F/(R*T))*eta6) - exp(-0.5*(F/(R*T))*eta6))
#print(U2_ref)
#print(i1)
#print(i1.subs(phi1, 2.4).subs(phi2, 0))

# =============================================================================
# # For verification purposes:
# i1 = i01
# i2 = i02
# i3 = i03
# i4 = i04
# i5 = i05
# i6 = i06
# =============================================================================

# =============================================================================
# ## Now Define the functions for seperator and cathode regions ##
# =============================================================================

# =============================================================================
# ## u1_cath for phi1 at cathode ##
# =============================================================================
N_Li_cath = -((z_Li*D_Li_cath*F)/(R*T))*(C_Li)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s8_cath = -((z_s8*D_s8_cath*F)/(R*T))*(C_s8)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s8_2_cath = -((z_s8_2*D_s8_2_cath*F)/(R*T))*(C_s8_2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s6_cath = -((z_s6*D_s6_cath*F)/(R*T))*(C_s6)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s4_cath = -((z_s4*D_s4_cath*F)/(R*T))*(C_s4)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s2_cath = -((z_s2*D_s2_cath*F)/(R*T))*(C_s2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s_cath = -((z_s*D_s_cath*F)/(R*T))*(C_s)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_A_cath = -((z_A*D_A_cath*F)/(R*T))*(C_A)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
div_flux_sum_cath = F*(z_Li*N_Li_cath + z_s8*N_s8_cath + z_s8_2*N_s8_2_cath + z_s6*N_s6_cath + z_s4*N_s4_cath + z_s2*N_s2_cath + z_s*N_s_cath + z_A*N_A_cath)
LHS_u1_cath = div_flux_sum_cath
RHS_u1_cath = sigma*(phi1_postx - 2*phi1 + phi1_prevx)/(dx**2)
u1_time = (phi1 - phi1_prevt)/dt
u1_cath = u1_time - (LHS_u1_cath - RHS_u1_cath)
#print(diff(u1_cath, phi2_postx))
#print('\n')

# =============================================================================
# ## u2_cath for phi2 at cathode ##
# =============================================================================
LHS_u2_cath = div_flux_sum_cath
RHS_u2_cath = av*(i2 + i3 + i4 + i5 + i6)
u2_time = (phi2 - phi2_prevt)/dt
u2_cath = u2_time - (LHS_u2_cath - RHS_u2_cath)
#print(u2_cath.subs(phi1, 2.4).subs(phi2, 0).subs(phi2_postx, 0).subs(phi2_prevx, 0))
#print(diff(u2, phi2_postx))
#print('\n')

# =============================================================================
# ## u1_sep for phi1 at seperator ##
# =============================================================================
N_Li_sep = -((z_Li*D_Li_sep*F)/(R*T))*(C_Li)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s8_sep = -((z_s8*D_s8_sep*F)/(R*T))*(C_s8)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s8_2_sep = -((z_s8_2*D_s8_2_sep*F)/(R*T))*(C_s8_2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s6_sep = -((z_s6*D_s6_sep*F)/(R*T))*(C_s6)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s4_sep = -((z_s4*D_s4_sep*F)/(R*T))*(C_s4)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s2_sep = -((z_s2*D_s2_sep*F)/(R*T))*(C_s2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_s_sep = -((z_s*D_s_sep*F)/(R*T))*(C_s)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
N_A_sep = -((z_A*D_A_sep*F)/(R*T))*(C_A)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2)
div_flux_sum_sep = F*(z_Li*N_Li_sep + z_s8*N_s8_sep + z_s8_2*N_s8_2_sep + z_s6*N_s6_sep + z_s4*N_s4_sep + z_s2*N_s2_sep + z_s*N_s_sep + z_A*N_A_sep)
LHS_u1_sep = div_flux_sum_sep
RHS_u1_sep = 0*(phi1_postx - 2*phi1 + phi1_prevx)/(dx**2) ## No sigma in seperator region
u1_time = (phi1 - phi1_prevt)/dt
u1_sep = u1_time - (LHS_u1_sep - RHS_u1_sep)
#print(diff(u1, phi2_postx))
#print('\n')

# =============================================================================
# ## u2_sep for phi2 at seperator ##
# =============================================================================
LHS_u2_sep = div_flux_sum_sep
RHS_u2_sep = 0*(i2 + i3 + i4 + i5 + i6) ## No partial currents in seperator region
u2_time = (phi2 - phi2_prevt)/dt
u2_sep = u2_time - (LHS_u2_sep - RHS_u2_sep)
#print(diff(u2, phi2_postx))
#print('\n')

## Overall u-function-arrays
u_list_cath = [u1_cath, u2_cath]
u_list_sep = [u1_sep, u2_sep]

variables = [phi1, phi2]

# =============================================================================
# ## Form Jacobian Matrix ##
# =============================================================================
num_var = len(variables) ## Number of variables (w/o spatial discretisation)
num_space = len(x_space) ## Number of spatially discretised points

n = num_var*num_space # Number of variables to solve for 
jacob = []
for i in range(n):
    jacob.append([])
    for j in range(n):
        jacob[i].append(0)

num_rows, num_cols = np.shape(jacob)

## Function to create dict and row labels
def labels(var_list, num_space, jacob_shape):
    varlist = []
    for i in range(len(var_list)):
        temp = []
        for j in range(num_space):
            to_append = (var_list[i], j)
            temp.append(to_append)
            
        varlist.append(temp)
    
    varlist = [item for sublist in varlist for item in sublist]
    #return(varlist)
    ## Now we create the dictionary label and row_label based of jacobian shape
    ## Label each internal points with respective dependent variables
    # Initialize an empty dictionary
    label_dict = {}

    # Get the shape of the matrix
    num_rows, num_cols = jacob_shape

    # Iterate through the matrix and assign labels to each position in the dictionary
    for i in range(num_rows):
        for j in range(num_cols):
            label = varlist[j]
            position = (i, j)
            label_dict[position] = label
         
    ## Another row label
    row_label = {}
    for i in range(num_rows):
        label = varlist[i]
        row = i
        row_label[row] = label
        
    return(label_dict, row_label)


label_dict, row_label = labels(variables, num_space, np.shape(jacob))

# =============================================================================
# Define function array (This can be used to define different 
#                        functions for seperator and cathode regions)
# =============================================================================
u_func_array = []
#u_list_sep_str = ['u1_sep', 'u2_sep'] ## string array for testing
#u_list_cath_str = ['u1_cath', 'u2_cath'] ## string array for testing

## Create a list without boundary functions 1st
for i in range(num_var):
    temp_holder = []
    for j in range(num_space):
        if j < index_sep_cath_boundary:
            temp_holder.append(u_list_sep[i])
        if j == index_sep_cath_boundary:
            temp_holder.append(0)
        if j > index_sep_cath_boundary:
            temp_holder.append(u_list_cath[i])

    u_func_array.append(temp_holder)
    
for i in range(num_var):
    for j in range(num_space):
        if j == 0:
            u_func_array[i][j] = 0
        if j == num_space-1:
            u_func_array[i][j] = 0
 
u_func_array_for_later = u_func_array.copy()
u_func_array = [item for sublist in u_func_array for item in sublist] ## Flatten u_array

# =============================================================================
# Remove phi1 dependancies in separator region (Via Penalty Method)
# =============================================================================
## Identify the index of the points of the separator
index_separator_points = []
for i in range(1, index_sep_cath_boundary):
    index_separator_points.append(i) ## These values corresponds to the index of points in the separator region

## Identify the indices that are to be deleted
index_to_delete = []
for i in range(num_rows):
    if row_label[i][0] == phi1 and row_label[i][1] in index_separator_points:
        index_to_delete.append(i)

# =============================================================================
# ## Iterate through jacobian matrix and fill in points based on labels (Non_Boundary Points)
# =============================================================================
k = 0
#print(row_label)
for i in range(num_rows):
    for j in range(num_cols):
        if row_label[i][1] == 0 or row_label[i][1] == num_space-1 or row_label[i][1] == index_sep_cath_boundary: ## Can be generalized when knowing the number of spatially dicretized points
            if row_label[i][1] == 0:
                k += 1
            break
        
        ## Handle for general variables ##
        if i == j:
            syms_in_func = [sym for sym in u_func_array[i].free_symbols if sym in syms_to_solve]
            modified_symbols = [str(sym).replace('_prevx', '').replace('_postx', '') for sym in syms_in_func]
            unique_modified_symbols = list(dict.fromkeys(modified_symbols))
            syms_unspace = [symbols(item) for item in unique_modified_symbols]
            for item in syms_unspace:
                search = (item, i-num_space*(k-1))
                for key, value in row_label.items():
                    if search == (value):
                        column = key
                        break
                
                prevx_item = symbols(item.name + '_prevx')
                postx_item = symbols(item.name + '_postx')
                jacob[i][column] = diff(u_func_array[i], item)
                if prevx_item in syms_in_func:
                    jacob[i][column-1] = diff(u_func_array[i], prevx_item)
                if postx_item in syms_in_func:
                    jacob[i][column+1] = diff(u_func_array[i], postx_item)

## Introduce the penalty method
for i in range(num_rows):
    for j in range(num_cols):
        if i == j and i in index_to_delete:
            jacob[i][j] = jacob[i][j] + 1e50 ## Add a large penalty factor
            
#print(jacob)

# =============================================================================
# =============================================================================
# =============================================================================
# Now we define the boundary condition equations
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# At x=0 (Anode/Seperator interface) boundary conditions
# =============================================================================
## Boundary condition for phi1 = 0 at x=0
ub1_x0 = phi1
ub2_x0 = phi2_postx - phi2 + dx*((i1*R*T)/(F*F*z_Li*D_Li_sep*C_Li))

ub_x0_list = [ub1_x0, ub2_x0]

# =============================================================================
# At x=Ls (Seperator/Cathode interface) boundary conditions 
# =============================================================================
#ub1_xs = -sigma*((phi1_postx - phi1_prevx)/(2*dx))
ub1_xs = -sigma*((phi1_postx - phi1)/(dx))
## Flux sum at seperator section
N_Li_sep_xs = -((z_Li*D_Li_sep*F)/(R*T))*(C_Li)*(phi2 - phi2_prevx)/(dx)
N_s8_sep_xs = -((z_s8*D_s8_sep*F)/(R*T))*(C_s8)*(phi2 - phi2_prevx)/(dx)
N_s8_2_sep_xs = -((z_s8_2*D_s8_2_sep*F)/(R*T))*(C_s8_2)*(phi2 - phi2_prevx)/(dx)
N_s6_sep_xs = -((z_s6*D_s6_sep*F)/(R*T))*(C_s6)*(phi2 - phi2_prevx)/(dx)
N_s4_sep_xs = -((z_s4*D_s4_sep*F)/(R*T))*(C_s4)*(phi2 - phi2_prevx)/(dx)
N_s2_sep_xs = -((z_s2*D_s2_sep*F)/(R*T))*(C_s2)*(phi2 - phi2_prevx)/(dx)
N_s_sep_xs = -((z_s*D_s_sep*F)/(R*T))*(C_s)*(phi2 - phi2_prevx)/(dx)
N_A_sep_xs = -((z_A*D_A_sep*F)/(R*T))*(C_A)*(phi2 - phi2_prevx)/(dx)
flux_sum_sep_xs = F*(z_Li*N_Li_sep_xs + z_s8*N_s8_sep_xs + z_s8_2*N_s8_2_sep_xs + z_s6*N_s6_sep_xs + z_s4*N_s4_sep_xs + z_s2*N_s2_sep_xs + z_s*N_s_sep_xs + z_A*N_A_sep_xs)
## Flux sum at cathode section
N_Li_cath_xs = -((z_Li*D_Li_cath*F)/(R*T))*(C_Li)*(phi2_postx - phi2)/(dx)
N_s8_cath_xs = -((z_s8*D_s8_cath*F)/(R*T))*(C_s8)*(phi2_postx - phi2)/(dx)
N_s8_2_cath_xs = -((z_s8_2*D_s8_2_cath*F)/(R*T))*(C_s8_2)*(phi2_postx - phi2)/(dx)
N_s6_cath_xs = -((z_s6*D_s6_cath*F)/(R*T))*(C_s6)*(phi2_postx - phi2)/(dx)
N_s4_cath_xs = -((z_s4*D_s4_cath*F)/(R*T))*(C_s4)*(phi2_postx - phi2)/(dx)
N_s2_cath_xs = -((z_s2*D_s2_cath*F)/(R*T))*(C_s2)*(phi2_postx - phi2)/(dx)
N_s_cath_xs = -((z_s*D_s_cath*F)/(R*T))*(C_s)*(phi2_postx - phi2)/(dx)
N_A_cath_xs = -((z_A*D_A_cath*F)/(R*T))*(C_A)*(phi2_postx - phi2)/(dx)
flux_sum_cath_xs = F*(z_Li*N_Li_cath_xs + z_s8*N_s8_cath_xs + z_s8_2*N_s8_2_cath_xs + z_s6*N_s6_cath_xs + z_s4*N_s4_cath_xs + z_s2*N_s2_cath_xs + z_s*N_s_cath_xs + z_A*N_A_cath_xs)

ub2_xs = flux_sum_sep_xs - flux_sum_cath_xs

ub_xs_list = [ub1_xs, ub2_xs]

# =============================================================================
# At x=Ls+Lc (Cathode current collector) boundary conditions 
# =============================================================================
ub1_xL = phi1 - phi1_prevx + ((I_app*dx)/(sigma*A))

N_Li_cath_xL = -((z_Li*D_Li_cath*F)/(R*T))*(C_Li)*(phi2 - phi2_prevx)/(dx)
N_s8_cath_xL = -((z_s8*D_s8_cath*F)/(R*T))*(C_s8)*(phi2 - phi2_prevx)/(dx)
N_s8_2_cath_xL = -((z_s8_2*D_s8_2_cath*F)/(R*T))*(C_s8_2)*(phi2 - phi2_prevx)/(dx)
N_s6_cath_xL = -((z_s6*D_s6_cath*F)/(R*T))*(C_s6)*(phi2 - phi2_prevx)/(dx)
N_s4_cath_xL = -((z_s4*D_s4_cath*F)/(R*T))*(C_s4)*(phi2 - phi2_prevx)/(dx)
N_s2_cath_xL = -((z_s2*D_s2_cath*F)/(R*T))*(C_s2)*(phi2 - phi2_prevx)/(dx)
N_s_cath_xL = -((z_s*D_s_cath*F)/(R*T))*(C_s)*(phi2 - phi2_prevx)/(dx)
N_A_cath_xL = -((z_A*D_A_cath*F)/(R*T))*(C_A)*(phi2 - phi2_prevx)/(dx)
flux_sum_cath_xL = F*(z_Li*N_Li_cath_xL + z_s8*N_s8_cath_xL + z_s8_2*N_s8_2_cath_xL + z_s6*N_s6_cath_xL + z_s4*N_s4_cath_xL + z_s2*N_s2_cath_xL + z_s*N_s_cath_xL + z_A*N_A_cath_xL)
ub2_xL = flux_sum_cath_xL

ub_xL_list = [ub1_xL, ub2_xL]

# =============================================================================
# Update u_func_array
# =============================================================================
for i in range(num_var):
    for j in range(num_space):
        if j == 0:
            u_func_array_for_later[i][j] = ub_x0_list[i]
        if j == index_sep_cath_boundary:
            u_func_array_for_later[i][j] = ub_xs_list[i]
        if j == num_space-1:
            u_func_array_for_later[i][j] = ub_xL_list[i]

u_func_array = [item for sublist in u_func_array_for_later for item in sublist] ## Flatten u_array

# =============================================================================
# Update Jacobian matrix
# =============================================================================
## Fill for x=0 boundary points 1st ##
k = 0
for i in range(num_rows):
    for j in range(num_cols):
        if row_label[i][1] == 0:
            if j == 0:
                k += 1
                
            if i == j:
                syms_in_func = [sym for sym in ub_x0_list[k-1].free_symbols if sym in syms_to_solve]
                modified_symbols = [str(sym).replace('_prevx', '').replace('_postx', '') for sym in syms_in_func]
                unique_modified_symbols = list(dict.fromkeys(modified_symbols))
                syms_unspace = [symbols(item) for item in unique_modified_symbols]
                for item in syms_unspace:
                    search = (item, i-num_space*(k-1))
                    for key, value in row_label.items():
                        if search == (value):
                            column = key
                            break
                    
                    prevx_item = symbols(item.name + '_prevx')
                    postx_item = symbols(item.name + '_postx')
                    if item in syms_in_func:
                        jacob[i][column] = diff(ub_x0_list[k-1], item)
                    if postx_item in syms_in_func:
                        jacob[i][column+1] = diff(ub_x0_list[k-1], postx_item)
                    if prevx_item in syms_in_func:
                        jacob[i][column-1] = diff(ub_x0_list[k-1], prevx_item)
                        
## Fill for x=Ls boundary points
k = 0
for i in range(num_rows):
    for j in range(num_cols):
        if row_label[i][1] == index_sep_cath_boundary:
            if j == 0:
                k += 1
                
            if i == j:
                syms_in_func = [sym for sym in ub_xs_list[k-1].free_symbols if sym in syms_to_solve]
                modified_symbols = [str(sym).replace('_prevx', '').replace('_postx', '') for sym in syms_in_func]
                unique_modified_symbols = list(dict.fromkeys(modified_symbols))
                syms_unspace = [symbols(item) for item in unique_modified_symbols]
                for item in syms_unspace:
                    search = (item, i-num_space*(k-1))
                    for key, value in row_label.items():
                        if search == (value):
                            column = key
                            break
                    
                    prevx_item = symbols(item.name + '_prevx')
                    postx_item = symbols(item.name + '_postx')
                    if item in syms_in_func:
                        jacob[i][column] = diff(ub_xs_list[k-1], item)
                    if postx_item in syms_in_func:
                        jacob[i][column+1] = diff(ub_xs_list[k-1], postx_item)
                    if prevx_item in syms_in_func:
                        jacob[i][column-1] = diff(ub_xs_list[k-1], prevx_item)
                        
## Fill for x=Ls+Lc boundary points
k = 0
for i in range(num_rows):
    for j in range(num_cols):
        if row_label[i][1] == num_space-1:
            if j == 0:
                k += 1
                
            if i == j:
                syms_in_func = [sym for sym in ub_xL_list[k-1].free_symbols if sym in syms_to_solve]
                modified_symbols = [str(sym).replace('_prevx', '').replace('_postx', '') for sym in syms_in_func]
                unique_modified_symbols = list(dict.fromkeys(modified_symbols))
                syms_unspace = [symbols(item) for item in unique_modified_symbols]
                for item in syms_unspace:
                    search = (item, i-num_space*(k-1))
                    for key, value in row_label.items():
                        if search == (value):
                            column = key
                            break
                    
                    prevx_item = symbols(item.name + '_prevx')
                    postx_item = symbols(item.name + '_postx')
                    if item in syms_in_func:
                        jacob[i][column] = diff(ub_xL_list[k-1], item)
                    if postx_item in syms_in_func:
                        jacob[i][column+1] = diff(ub_xL_list[k-1], postx_item)
                    if prevx_item in syms_in_func:
                        jacob[i][column-1] = diff(ub_xL_list[k-1], prevx_item)

# =============================================================================
# Check Matrix Sparsity         
# =============================================================================
to_check = [element for sublist in jacob for element in sublist]
no_zeros = 0
for i in range(len(to_check)):
    if to_check[i] == 0:
        no_zeros += 1
        
percentage_sparsity = 100*(no_zeros/len(to_check))
print('% Sparsity of Potentials Jacobian Matrix: ', percentage_sparsity)

# =============================================================================
# Lambdify both the u_func_array and jacob
# =============================================================================
## For loop to lambdify the Jacobian functions
print('Lambdifying Potentials Jacobian Matrix:')
count_jacob = 0
for i in range(num_rows):
    for j in range(num_cols):
        if jacob[i][j] != 0:
            jacob[i][j] = lambdify(syms, jacob[i][j], 'numpy')
            count_jacob += 1
            update_loading_bar(count_jacob, ((num_var*num_space)**2 - no_zeros))

print()
print('Potentials Jacobian Matrix Lambdification Complete !!!')

## For loop to lambdify the array functions
print('Lambdifying Potentials Function Arrays:')
count_u = 0
for i in range(len(u_func_array)):
    u_func_array[i] = lambdify(syms, u_func_array[i], 'numpy')
    count_u += 1
    update_loading_bar(count_u, num_var*num_space)

print()
print('Potentials Function Arrays Lambdification Complete !!!')  

## Now we make a model class to carry out substitutions and solver function to solve ##
## These will be done in another script for ease of sub-dividing scripts ##