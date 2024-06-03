from sympy import symbols, diff, exp, sqrt, log, lambdify, pprint, Abs
import numpy as np
import timeit
import func_potentials as func

## Loading bar function:
def update_loading_bar(iteration, total):
    percent_complete = (iteration / total) * 100
    loading_bar = f"[{'#' * int(percent_complete / 2)}{' ' * (50 - int(percent_complete / 2))}] {percent_complete:.2f}%"
    print('\033[K' + loading_bar, end='\r')  # \033[K clears the line

# =============================================================================
# THIS SCRIPT IS TO DEFINE THE EQUATIONS AND PDEs DISCRETIZATION FOR THE
# FULL (SEPARATOR + CATHODE) 1D Li-S MODEL
# =============================================================================

## This model is the full 6-stage seperator + cathode model (**Added Anion Salt, An- species) ##
## Includes non-constant porosity and precipitation/dissolution dynamics ##

## Define all parameters as symbols

F = 96485.3321233100184
R = 8.3145
T = 292.15
L_cath = func.L_cath
L_sep = func.L_sep
A = 0.28
a0 = 132762 ## Redfine as a0 here define av later
e_cath_init = 0.7 ## Initial cathode porosity
sigma = 1
i01 = 0.394 #0.5
i02 = 1.972
i03 = 0.019
i04 = 0.019
i05 = 1.97e-4
i06 = 1.97e-7 #2e-9
U1 = 0
U2 = 2.39 #2.35 #2.41
U3 = 2.37 #2.06 #2.35
U4 = 2.24 #2.05 #2.23
U5 = 2.04
U6 = 2.01
z_Li = 1
z_s8 = 0
z_s8_2 = -2
z_s6 = -2
z_s4 = -2
z_s2 = -2
z_s1 = -2
z_An = -1
## Precipitation/Dissolution coefficients
k_s8 = 1 #5 #1 #0 #(0 for no precipitation/dissolution)
k_Li2s = 6.875e-5 #3.45e-5 #100 #27.5 #0 #(0 for no precipitation/dissolution)
ksp_s8 = 19
ksp_Li2s = 9.95e-4 #9.95e3 #9.95e-4 * 4e8 #1e2 #3e-5 #1.55e7            
V_s8 = 1.239e-4
V_Li2s = 2.768e-5 #2.4e-5
## These reference values are the initial conditions for the species ##
C_Li = 1001
C_s8 = 19
C_s8_2 = 0.18
C_s6 = 0.32
C_s4 = 0.02
C_s2 = 5.23e-7
C_s1 = 8.27e-10
C_An = 1000 
I_app = symbols('I_app') #0.394*A ## (NOTE: '-' for charge '+' for discharge)
# =============================================================================
# The disffusion coefficients defined are bulk coefficients
# =============================================================================
## Diffusion coefficients (bulk):
DD = 1
D_Li = 1e-10*DD #0.88e-12
D_s8 = 1e-9*DD #0.88e-11
D_s8_2 = 1e-10*DD #3.5e-12
D_s6 = 6e-10*DD #3.5e-12
D_s4 = 1e-10*DD #1.75e-12
D_s2 = 1e-10*DD #0.88e-12
D_s1 = 1e-10*DD #0.88e-12
D_An = 4e-10*DD #3.5e-12

## Space discretisation
no_points = func.no_points ## 28 Points here is 50 discretized points in space
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
Li = symbols('Li')
s8 = symbols('s8')
s8_2 = symbols('s8_2')
s6 = symbols('s6')
s4 = symbols('s4')
s2 = symbols('s2')
s1 = symbols('s1')
An = symbols('An') # Anion salt
e_cath = symbols('e_cath') ## Cathode porosity
e_sep = symbols('e_sep') ## Separator porosity
e_s8_cath = symbols('e_s8_cath') ## s8 cathode volume fraction
e_s8_sep = symbols('e_s8_sep') ## s8 separator volume fraction
e_Li2s_cath = symbols('e_Li2s_cath') ## Li2s cathode volume fraction
e_Li2s_sep = symbols('e_Li2s_sep') ## Li2s separator volume fraction
phi1 = symbols('phi1') # Solid state potential
phi2 = symbols('phi2') # Electrolyte potential

## 2) Current in time before in space
Li_prevx = symbols('Li_prevx')
s8_prevx = symbols('s8_prevx')
s8_2_prevx = symbols('s8_2_prevx')
s6_prevx = symbols('s6_prevx')
s4_prevx = symbols('s4_prevx')
s2_prevx = symbols('s2_prevx')
s1_prevx = symbols('s1_prevx')
An_prevx = symbols('An_prevx') # Anion salt
e_cath_prevx = symbols('e_cath_prevx') ## Cathode porosity
e_sep_prevx = symbols('e_sep_prevx') ## Separator porosity
e_s8_cath_prevx = symbols('e_s8_cath_prevx') ## s8 cathode volume fraction
e_s8_sep_prevx = symbols('e_s8_sep_prevx') ## s8 separator volume fraction
e_Li2s_cath_prevx = symbols('e_Li2s_cath_prevx') ## Li2s cathode volume fraction
e_Li2s_sep_prevx = symbols('e_Li2s_sep_prevx') ## Li2s separator volume fraction
phi1_prevx = symbols('phi1_prevx') # Solid state potential
phi2_prevx = symbols('phi2_prevx') # Electrolyte potential

## 3) Current in time after in space
Li_postx = symbols('Li_postx')
s8_postx = symbols('s8_postx')
s8_2_postx = symbols('s8_2_postx')
s6_postx = symbols('s6_postx')
s4_postx = symbols('s4_postx')
s2_postx = symbols('s2_postx')
s1_postx = symbols('s1_postx')
An_postx = symbols('An_postx') # Anion salt
e_cath_postx = symbols('e_cath_postx') ## Cathode porosity
e_sep_postx = symbols('e_sep_postx') ## Separator porosity
e_s8_cath_postx = symbols('e_s8_cath_postx') ## s8 cathode volume fraction
e_s8_sep_postx = symbols('e_s8_sep_postx') ## s8 separator volume fraction
e_Li2s_cath_postx = symbols('e_Li2s_cath_postx') ## Li2s cathode volume fraction
e_Li2s_sep_postx = symbols('e_Li2s_sep_postx') ## Li2s separator volume fraction
phi1_postx = symbols('phi1_postx') # Solid state potential
phi2_postx = symbols('phi2_postx') # Electrolyte potential

## 4) Previous in time current in space
Li_prevt = symbols('Li_prevt')
s8_prevt = symbols('s8_prevt')
s8_2_prevt = symbols('s8_2_prevt')
s6_prevt = symbols('s6_prevt')
s4_prevt = symbols('s4_prevt')
s2_prevt = symbols('s2_prevt')
s1_prevt = symbols('s1_prevt')
An_prevt = symbols('An_prevt') # Anion salt
e_cath_prevt = symbols('e_cath_prevt') ## Cathode porosity
e_sep_prevt = symbols('e_sep_prevt') ## Separator porosity
e_s8_cath_prevt = symbols('e_s8_cath_prevt') ## s8 cathode volume fraction
e_s8_sep_prevt = symbols('e_s8_sep_prevt') ## s8 separator volume fraction
e_Li2s_cath_prevt = symbols('e_Li2s_cath_prevt') ## Li2s cathode volume fraction
e_Li2s_sep_prevt = symbols('e_Li2s_sep_prevt') ## Li2s separator volume fraction
phi1_prevt = symbols('phi1_prevt') # Solid state potential
phi2_prevt = symbols('phi2_prevt') # Electrolyte potential

## Introduce lambdify symbols
Li_syms = [Li_prevt, Li_prevx, Li, Li_postx]
s8_syms = [s8_prevt, s8_prevx, s8, s8_postx]
s8_2_syms = [s8_2_prevt, s8_2_prevx, s8_2, s8_2_postx]
s6_syms = [s6_prevt, s6_prevx, s6, s6_postx]
s4_syms = [s4_prevt, s4_prevx, s4, s4_postx]
s2_syms = [s2_prevt, s2_prevx, s2, s2_postx]
s1_syms = [s1_prevt, s1_prevx, s1, s1_postx]
An_syms = [An_prevt, An_prevx, An, An_postx]
e_cath_syms = [e_cath_prevt, e_cath_prevx, e_cath, e_cath_postx]
e_sep_syms = [e_sep_prevt, e_sep_prevx, e_sep, e_sep_postx]
e_s8_cath_syms = [e_s8_cath_prevt, e_s8_cath_prevx, e_s8_cath, e_s8_cath_postx]
e_s8_sep_syms = [e_s8_sep_prevt, e_s8_sep_prevx, e_s8_sep, e_s8_sep_postx]
e_Li2s_cath_syms = [e_Li2s_cath_prevt, e_Li2s_cath_prevx, e_Li2s_cath, e_Li2s_cath_postx]
e_Li2s_sep_syms = [e_Li2s_sep_prevt, e_Li2s_sep_prevx, e_Li2s_sep, e_Li2s_sep_postx]
phi1_syms = [phi1_prevt, phi1_prevx, phi1, phi1_postx]
phi2_syms = [phi2_prevt, phi2_prevx, phi2, phi2_postx]

## Symbols to solve for
Li_syms_to_solve = [Li_prevx, Li, Li_postx]
s8_syms_to_solve = [s8_prevx, s8, s8_postx]
s8_2_syms_to_solve = [s8_2_prevx, s8_2, s8_2_postx]
s6_syms_to_solve = [s6_prevx, s6, s6_postx]
s4_syms_to_solve = [s4_prevx, s4, s4_postx]
s2_syms_to_solve = [s2_prevx, s2, s2_postx]
s1_syms_to_solve = [s1_prevx, s1, s1_postx]
An_syms_to_solve = [An_prevx, An, An_postx]
e_cath_syms_to_solve = [e_cath_prevx, e_cath, e_cath_postx]
e_sep_syms_to_solve = [e_sep_prevx, e_sep, e_sep_postx]
e_s8_cath_syms_to_solve = [e_s8_cath_prevx, e_s8_cath, e_s8_cath_postx]
e_s8_sep_syms_to_solve = [e_s8_sep_prevx, e_s8_sep, e_s8_sep_postx]
e_Li2s_cath_syms_to_solve = [e_Li2s_cath_prevx, e_Li2s_cath, e_Li2s_cath_postx]
e_Li2s_sep_syms_to_solve = [e_Li2s_sep_prevx, e_Li2s_sep, e_Li2s_sep_postx]
phi1_syms_to_solve = [phi1_prevx, phi1, phi1_postx]
phi2_syms_to_solve = [phi2_prevx, phi2, phi2_postx]

syms = Li_syms + s8_syms + s8_2_syms + s6_syms + s4_syms + s2_syms + \
        s1_syms + An_syms + e_cath_syms + e_sep_syms + e_s8_cath_syms + e_s8_sep_syms + \
        e_Li2s_cath_syms + e_Li2s_sep_syms + phi1_syms + phi2_syms + [dx, dt, I_app] ## For lambdification

syms_to_solve = Li_syms_to_solve + s8_syms_to_solve + s8_2_syms_to_solve + s6_syms_to_solve + \
    s4_syms_to_solve + s2_syms_to_solve + s1_syms_to_solve + An_syms_to_solve + \
    e_cath_syms_to_solve + e_sep_syms_to_solve + e_s8_cath_syms_to_solve + e_s8_sep_syms_to_solve + \
    e_Li2s_cath_syms_to_solve + e_Li2s_sep_syms_to_solve + \
    phi1_syms_to_solve + phi2_syms_to_solve ## For Jacobian generalization

# =============================================================================
# ## Define dependant equations ##
# =============================================================================
## Reference reaction voltage
#U1_ref = U1 - ((R*T)/(F))*(-1*log(C_Li/1e3))
U1_ref = U1
U2_ref = U2 - ((R*T)/(2*F))*log(C_s8_2/C_s8)
U3_ref = U3 - ((R*T)/(F))*(-1.5*(log(C_s8_2/1e3)) + 2*(log(C_s6/1e3)))
U4_ref = U4 - ((R*T)/(F))*(-1*log(C_s6/1e3) + 1.5*log(C_s4/1e3))
U5_ref = U5 - ((R*T)/(F))*(-0.5*log(C_s4/1e3) + log(C_s2/1e3))
U6_ref = U6 - ((R*T)/(F))*(-0.5*log(C_s2/1e3) + log(C_s1/1e3))
## Nernst potentials for each reaction
eta1 = phi1 - phi2 - U1_ref
eta2 = phi1 - phi2 - U2_ref
eta3 = phi1 - phi2 - U3_ref
eta4 = phi1 - phi2 - U4_ref
eta5 = phi1 - phi2 - U5_ref
eta6 = phi1 - phi2 - U6_ref

## Reaction partial currents
i1 = i01*(exp(0.5*(F/(R*T))*eta1) - (Li/C_Li)*exp(-0.5*(F/(R*T))*eta1)) 
i2 = i02*(((s8_2/C_s8_2)**0.5)*exp(0.5*(F/(R*T))*eta2) - ((s8/C_s8)**0.5)*exp(-0.5*(F/(R*T))*eta2))
i3 = i03*(((s6/C_s6)**2)*exp(0.5*(F/(R*T))*eta3) - ((s8_2/C_s8_2)**1.5)*exp(-0.5*(F/(R*T))*eta3))
i4 = i04*(((s4/C_s4)**1.5)*exp(0.5*(F/(R*T))*eta4) - (s6/C_s6)*exp(-0.5*(F/(R*T))*eta4))
i5 = i05*((s2/C_s2)*exp(0.5*(F/(R*T))*eta5) - ((s4/C_s4)**0.5)*exp(-0.5*(F/(R*T))*eta5))
i6 = i06*((s1/C_s1)*exp(0.5*(F/(R*T))*eta6) - ((s2/C_s2)**0.5)*exp(-0.5*(F/(R*T))*eta6))

## Species electrochemical reaction equations
av = a0*((e_cath/e_cath_init)**1.5) ## Dynamic definition of specific area of cathode
r_Li = av*(i1/F)
r_s8 = av*(i2/(2*F))
r_s8_2 = av*(1.5*(i3/F) - 0.5*(i2/F))
r_s6 = av*((i4/F) - 2*(i3/F))
r_s4 = av*(0.5*(i5/F) - 1.5*(i4/F))
r_s2 = av*(0.5*(i6/F) - (i5/F))
r_s1 = av*(-1*(i6/F))

## Define Precipitation/Dissolution rate equations for species Li_+, s8, s1_2- (cathode & separator)
R_Li_cath = 2*(k_Li2s*e_Li2s_cath*((Li**2)*(s1) - ksp_Li2s))
R_Li_sep = 2*(k_Li2s*e_Li2s_sep*((Li**2)*(s1) - ksp_Li2s))
R_s8_cath = k_s8*e_s8_cath*(s8 - ksp_s8)
R_s8_sep = k_s8*e_s8_sep*(s8 - ksp_s8)
R_s1_cath = 1*(k_Li2s*e_Li2s_cath*((Li**2)*(s1) - ksp_Li2s))
R_s1_sep = 1*(k_Li2s*e_Li2s_sep*((Li**2)*(s1) - ksp_Li2s))

## Define cathode and separator diffusion coefficients
## Dynamic coefficients at cathode:
D_Li_cath = D_Li*(e_cath**1.5)
D_s8_cath = D_s8*(e_cath**1.5)
D_s8_2_cath = D_s8_2*(e_cath**1.5)
D_s6_cath = D_s6*(e_cath**1.5)
D_s4_cath = D_s4*(e_cath**1.5)
D_s2_cath = D_s2*(e_cath**1.5)
D_s1_cath = D_s1*(e_cath**1.5)
D_An_cath = D_An*(e_cath**1.5)
## Dynamic coefficients at seperator:
D_Li_sep = D_Li*(e_sep**1.5)
D_s8_sep = D_s8*(e_sep**1.5)
D_s8_2_sep = D_s8_2*(e_sep**1.5)
D_s6_sep = D_s6*(e_sep**1.5)
D_s4_sep = D_s4*(e_sep**1.5)
D_s2_sep = D_s2*(e_sep**1.5)
D_s1_sep = D_s1*(e_sep**1.5)
D_An_sep = D_An*(e_sep**1.5)

# =============================================================================
# ## Now Define the functions for seperator and cathode regions ##
# ## Re-defined for non-constant cathode and separator porosity ##
# =============================================================================
## Definition of cathode div flux terms for each species
div_N_Li_cath = -(D_Li_cath*(Li_postx - 2*Li + Li_prevx)/(dx**2) + (Li_postx - Li)*(1.5*D_Li*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_Li*F)/(R*T))*((D_Li_cath)*(Li)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_Li_cath)*(Li_postx - Li)*(phi2_postx - phi2)/(dx**2) + (Li)*(phi2_postx - phi2)*(1.5*D_Li*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_s8_cath = -(D_s8_cath*(s8_postx - 2*s8 + s8_prevx)/(dx**2) + (s8_postx - s8)*(1.5*D_s8*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_s8*F)/(R*T))*((D_s8_cath)*(s8)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s8_cath)*(s8_postx - s8)*(phi2_postx - phi2)/(dx**2) + (s8)*(phi2_postx - phi2)*(1.5*D_s8*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_s8_2_cath = -(D_s8_2_cath*(s8_2_postx - 2*s8_2 + s8_2_prevx)/(dx**2) + (s8_2_postx - s8_2)*(1.5*D_s8_2*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_s8_2*F)/(R*T))*((D_s8_2_cath)*(s8_2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s8_2_cath)*(s8_2_postx - s8_2)*(phi2_postx - phi2)/(dx**2) + (s8_2)*(phi2_postx - phi2)*(1.5*D_s8_2*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_s6_cath = -(D_s6_cath*(s6_postx - 2*s6 + s6_prevx)/(dx**2) + (s6_postx - s6)*(1.5*D_s6*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_s6*F)/(R*T))*((D_s6_cath)*(s6)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s6_cath)*(s6_postx - s6)*(phi2_postx - phi2)/(dx**2) + (s6)*(phi2_postx - phi2)*(1.5*D_s6*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_s4_cath = -(D_s4_cath*(s4_postx - 2*s4 + s4_prevx)/(dx**2) + (s4_postx - s4)*(1.5*D_s4*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_s4*F)/(R*T))*((D_s4_cath)*(s4)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s4_cath)*(s4_postx - s4)*(phi2_postx - phi2)/(dx**2) + (s4)*(phi2_postx - phi2)*(1.5*D_s4*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_s2_cath = -(D_s2_cath*(s2_postx - 2*s2 + s2_prevx)/(dx**2) + (s2_postx - s2)*(1.5*D_s2*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_s2*F)/(R*T))*((D_s2_cath)*(s2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s2_cath)*(s2_postx - s2)*(phi2_postx - phi2)/(dx**2) + (s2)*(phi2_postx - phi2)*(1.5*D_s2*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_s1_cath = -(D_s1_cath*(s1_postx - 2*s1 + s1_prevx)/(dx**2) + (s1_postx - s1)*(1.5*D_s1*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_s1*F)/(R*T))*((D_s1_cath)*(s1)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s1_cath)*(s1_postx - s1)*(phi2_postx - phi2)/(dx**2) + (s1)*(phi2_postx - phi2)*(1.5*D_s1*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))
div_N_An_cath = -(D_An_cath*(An_postx - 2*An + An_prevx)/(dx**2) + (An_postx - An)*(1.5*D_An*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2)) - ((z_An*F)/(R*T))*((D_An_cath)*(An)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_An_cath)*(An_postx - An)*(phi2_postx - phi2)/(dx**2) + (An)*(phi2_postx - phi2)*(1.5*D_An*(e_cath**0.5)*(e_cath_postx - e_cath))/(dx**2))

## Definition of separator div flux terms for each species
div_N_Li_sep = -(D_Li_sep*(Li_postx - 2*Li + Li_prevx)/(dx**2) + (Li_postx - Li)*(1.5*D_Li*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_Li*F)/(R*T))*((D_Li_sep)*(Li)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_Li_sep)*(Li_postx - Li)*(phi2_postx - phi2)/(dx**2) + (Li)*(phi2_postx - phi2)*(1.5*D_Li*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_s8_sep = -(D_s8_sep*(s8_postx - 2*s8 + s8_prevx)/(dx**2) + (s8_postx - s8)*(1.5*D_s8*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_s8*F)/(R*T))*((D_s8_sep)*(s8)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s8_sep)*(s8_postx - s8)*(phi2_postx - phi2)/(dx**2) + (s8)*(phi2_postx - phi2)*(1.5*D_s8*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_s8_2_sep = -(D_s8_2_sep*(s8_2_postx - 2*s8_2 + s8_2_prevx)/(dx**2) + (s8_2_postx - s8_2)*(1.5*D_s8_2*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_s8_2*F)/(R*T))*((D_s8_2_sep)*(s8_2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s8_2_sep)*(s8_2_postx - s8_2)*(phi2_postx - phi2)/(dx**2) + (s8_2)*(phi2_postx - phi2)*(1.5*D_s8_2*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_s6_sep = -(D_s6_sep*(s6_postx - 2*s6 + s6_prevx)/(dx**2) + (s6_postx - s6)*(1.5*D_s6*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_s6*F)/(R*T))*((D_s6_sep)*(s6)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s6_sep)*(s6_postx - s6)*(phi2_postx - phi2)/(dx**2) + (s6)*(phi2_postx - phi2)*(1.5*D_s6*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_s4_sep = -(D_s4_sep*(s4_postx - 2*s4 + s4_prevx)/(dx**2) + (s4_postx - s4)*(1.5*D_s4*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_s4*F)/(R*T))*((D_s4_sep)*(s4)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s4_sep)*(s4_postx - s4)*(phi2_postx - phi2)/(dx**2) + (s4)*(phi2_postx - phi2)*(1.5*D_s4*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_s2_sep = -(D_s2_sep*(s2_postx - 2*s2 + s2_prevx)/(dx**2) + (s2_postx - s2)*(1.5*D_s2*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_s2*F)/(R*T))*((D_s2_sep)*(s2)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s2_sep)*(s2_postx - s2)*(phi2_postx - phi2)/(dx**2) + (s2)*(phi2_postx - phi2)*(1.5*D_s2*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_s1_sep = -(D_s1_sep*(s1_postx - 2*s1 + s1_prevx)/(dx**2) + (s1_postx - s1)*(1.5*D_s1*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_s1*F)/(R*T))*((D_s1_sep)*(s1)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_s1_sep)*(s1_postx - s1)*(phi2_postx - phi2)/(dx**2) + (s1)*(phi2_postx - phi2)*(1.5*D_s1*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))
div_N_An_sep = -(D_An_sep*(An_postx - 2*An + An_prevx)/(dx**2) + (An_postx - An)*(1.5*D_An*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2)) - ((z_An*F)/(R*T))*((D_An_sep)*(An)*(phi2_postx - 2*phi2 + phi2_prevx)/(dx**2) + (D_An_sep)*(An_postx - An)*(phi2_postx - phi2)/(dx**2) + (An)*(phi2_postx - phi2)*(1.5*D_An*(e_sep**0.5)*(e_sep_postx - e_sep))/(dx**2))

# =============================================================================
# ## Define discretized PDEs ##
# =============================================================================
## Dependent porosity equation definition (for 2nd Order reform)
R_sep = -(V_s8*k_s8*e_s8_sep*(s8 - ksp_s8) + V_Li2s*k_Li2s*e_Li2s_sep*((Li**2)*(s1) - ksp_Li2s))
R_sep_prevt = -(V_s8*k_s8*e_s8_sep_prevt*(s8_prevt - ksp_s8) + V_Li2s*k_Li2s*e_Li2s_sep_prevt*((Li_prevt**2)*(s1_prevt) - ksp_Li2s))
R_cath = -(V_s8*k_s8*e_s8_cath*(s8 - ksp_s8) + V_Li2s*k_Li2s*e_Li2s_cath*((Li**2)*(s1) - ksp_Li2s))
R_cath_prevt = -(V_s8*k_s8*e_s8_cath_prevt*(s8_prevt - ksp_s8) + V_Li2s*k_Li2s*e_Li2s_cath_prevt*((Li_prevt**2)*(s1_prevt) - ksp_Li2s))
                            ## Cathode PDEs ##
# =============================================================================
# ## u1_cath for Li at cathode ##
# =============================================================================
LHS_u1_cath = e_cath*(Li - Li_prevt)/(dt) + Li*R_cath
RHS_u1_cath = -div_N_Li_cath - R_Li_cath
u1_cath = LHS_u1_cath - RHS_u1_cath
#print(diff(u1_cath, phi2_postx))

# =============================================================================
# ## u2_cath for s8 at cathode ##
# =============================================================================
LHS_u2_cath = e_cath*(s8 - s8_prevt)/(dt) + s8*R_cath
RHS_u2_cath = -div_N_s8_cath + r_s8 - R_s8_cath
u2_cath = LHS_u2_cath - RHS_u2_cath
#print(diff(u2_cath, phi2_postx))

# =============================================================================
# ## u3_cath for s8_2 at cathode ##
# =============================================================================
LHS_u3_cath = e_cath*(s8_2 - s8_2_prevt)/(dt) + s8_2*R_cath
RHS_u3_cath = -div_N_s8_2_cath + r_s8_2
u3_cath = LHS_u3_cath - RHS_u3_cath
#print(diff(u3, phi2_postx))

# =============================================================================
# ## u4_cath for s6 at cathode ##
# =============================================================================
LHS_u4_cath = e_cath*(s6 - s6_prevt)/(dt) + s6*R_cath
RHS_u4_cath = -div_N_s6_cath + r_s6
u4_cath = LHS_u4_cath - RHS_u4_cath
#print(diff(u4, phi2_postx))

# =============================================================================
# ## u5_cath for s4 at cathode ##
# =============================================================================
LHS_u5_cath = e_cath*(s4 - s4_prevt)/(dt) + s4*R_cath
RHS_u5_cath = -div_N_s4_cath + r_s4
u5_cath = LHS_u5_cath - RHS_u5_cath
#print(diff(u5, phi2_postx))

# =============================================================================
# ## u6_cath for s2 at cathode ##
# =============================================================================
LHS_u6_cath = e_cath*(s2 - s2_prevt)/(dt) + s2*R_cath
RHS_u6_cath = -div_N_s2_cath + r_s2
u6_cath = LHS_u6_cath - RHS_u6_cath
#print(diff(u6, phi2_postx))

# =============================================================================
# ## u7_cath for s1 at cathode ##
# =============================================================================
LHS_u7_cath = e_cath*(s1 - s1_prevt)/(dt) + s1*R_cath
RHS_u7_cath = -div_N_s1_cath + r_s1 - R_s1_cath
u7_cath = LHS_u7_cath - RHS_u7_cath
#print(diff(u7, phi2_postx))

# =============================================================================
# ## u8_cath for An at cathode ##
# =============================================================================
LHS_u8_cath = e_cath*(An - An_prevt)/(dt) + An*R_cath
RHS_u8_cath = -div_N_An_cath
u8_cath = LHS_u8_cath - RHS_u8_cath
#print(diff(u8, phi2_postx))

# =============================================================================
# ## u9_cath for e_cath ##
# =============================================================================
LHS_u9_cath = (e_cath - e_cath_prevt)/(dt)
#RHS_u9_cath = 0.5*(R_cath + R_cath_prevt)
RHS_u9_cath = R_cath
u9_cath = LHS_u9_cath - RHS_u9_cath

# =============================================================================
# ## u10_cath for e_sep ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u10_cath = (e_sep - e_sep_prevt)/(dt)
#RHS_u10_cath = 0.5*(R_sep + R_sep_prevt)
RHS_u10_cath = R_sep
u10_cath = LHS_u10_cath - RHS_u10_cath

# =============================================================================
# ## u11_cath for e_s8_cath ##
# =============================================================================
LHS_u11_cath = (e_s8_cath - e_s8_cath_prevt)/(dt)
RHS_u11_cath = V_s8*k_s8*e_s8_cath*(s8 - ksp_s8)
u11_cath = LHS_u11_cath - RHS_u11_cath

# =============================================================================
# ## u12_cath for e_s8_sep ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u12_cath = (e_s8_sep - e_s8_sep_prevt)/(dt)
RHS_u12_cath = V_s8*k_s8*e_s8_sep*(s8 - ksp_s8)
u12_cath = LHS_u12_cath - RHS_u12_cath

# =============================================================================
# ## u13_cath for e_Li2s_cath ##
# =============================================================================
LHS_u13_cath = (e_Li2s_cath - e_Li2s_cath_prevt)/(dt)
RHS_u13_cath = V_Li2s*k_Li2s*e_Li2s_cath*((Li**2)*(s1) - ksp_Li2s)
u13_cath = LHS_u13_cath - RHS_u13_cath

# =============================================================================
# ## u14_cath for e_Li2s_sep ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u14_cath = (e_Li2s_sep - e_Li2s_sep_prevt)/(dt)
RHS_u14_cath = V_Li2s*k_Li2s*e_Li2s_sep*((Li**2)*(s1) - ksp_Li2s)
u14_cath = LHS_u14_cath - RHS_u14_cath

# =============================================================================
# ## u15_cath for phi1 at cathode ##
# =============================================================================
LHS_u15_cath = (z_Li*div_N_Li_cath + z_s8*div_N_s8_cath + z_s8_2*div_N_s8_2_cath + z_s6*div_N_s6_cath + z_s4*div_N_s4_cath + z_s2*div_N_s2_cath + z_s1*div_N_s1_cath + z_An*div_N_An_cath)
LHS_u15_cath_redefined = (i2 + i3 + i4 + i5 + i6) ## i1 not present in cathode region
RHS_u15_cath = (sigma/av)*(phi1_postx - 2*phi1 + phi1_prevx)/(dx**2)
u15_cath = LHS_u15_cath_redefined - RHS_u15_cath

# =============================================================================
# ## u16_cath for phi2 at cathode ##
# =============================================================================
LHS_u16_cath = LHS_u15_cath
RHS_u16_cath = (av/F)*(i2 + i3 + i4 + i5 + i6) ## i1 not present in cathode region
u16_cath = LHS_u16_cath - RHS_u16_cath

                            ## Separator PDEs ##
# =============================================================================
# ## u1_sep for Li at separator ##
# =============================================================================
LHS_u1_sep = e_sep*(Li - Li_prevt)/(dt) + Li*R_sep
RHS_u1_sep = -div_N_Li_sep - R_Li_sep # No electrochemical reactions in separator r_i = 0
u1_sep = LHS_u1_sep - RHS_u1_sep
#print(diff(u1_sep, phi2_postx))

# =============================================================================
# ## u2_sep for s8 at separator ##
# =============================================================================
LHS_u2_sep = e_sep*(s8 - s8_prevt)/(dt) + s8*R_sep
RHS_u2_sep = -div_N_s8_sep - R_s8_sep # No electrochemical reactions in separator r_i = 0
u2_sep = LHS_u2_sep - RHS_u2_sep
#print(diff(u2_sep, phi2_postx))

# =============================================================================
# ## u3_sep for s8_2 at separator ##
# =============================================================================
LHS_u3_sep = e_sep*(s8_2 - s8_2_prevt)/(dt) + s8_2*R_sep
RHS_u3_sep = -div_N_s8_2_sep # No electrochemical reactions in separator r_i = 0
u3_sep = LHS_u3_sep - RHS_u3_sep
#print(diff(u3_sep, phi2_postx))

# =============================================================================
# ## u4_sep for s6 at separator ##
# =============================================================================
LHS_u4_sep = e_sep*(s6 - s6_prevt)/(dt) + s6*R_sep
RHS_u4_sep = -div_N_s6_sep # No electrochemical reactions in separator r_i = 0
u4_sep = LHS_u4_sep - RHS_u4_sep
#print(diff(u4_sep, phi2_postx))

# =============================================================================
# ## u5_sep for s4 at separator ##
# =============================================================================
LHS_u5_sep = e_sep*(s4 - s4_prevt)/(dt) + s4*R_sep
RHS_u5_sep = -div_N_s4_sep # No electrochemical reactions in separator r_i = 0
u5_sep = LHS_u5_sep - RHS_u5_sep
#print(diff(u5_sep, phi2_postx))

# =============================================================================
# ## u6_sep for s2 at separator ##
# =============================================================================
LHS_u6_sep = e_sep*(s2 - s2_prevt)/(dt) + s2*R_sep
RHS_u6_sep = -div_N_s2_sep # No electrochemical reactions in separator r_i = 0
u6_sep = LHS_u6_sep - RHS_u6_sep
#print(diff(u6_sep, phi2_postx))

# =============================================================================
# ## u7_sep for s1 at separator ##
# =============================================================================
LHS_u7_sep = e_sep*(s1 - s1_prevt)/(dt) + s1*R_sep
RHS_u7_sep = -div_N_s1_sep - R_s1_sep # No electrochemical reactions in separator r_i = 0
u7_sep = LHS_u7_sep - RHS_u7_sep
#print(diff(u7_sep, phi2_postx))

# =============================================================================
# ## u8_sep for An at separator ##
# =============================================================================
LHS_u8_sep = e_sep*(An - An_prevt)/(dt) + An*R_sep
RHS_u8_sep = -div_N_An_sep # No electrochemical reactions in separator r_i = 0
u8_sep = LHS_u8_sep - RHS_u8_sep
#print(diff(u8_sep, phi2_postx))

# =============================================================================
# ## u9_sep for e_cath ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u9_sep = (e_cath - e_cath_prevt)/(dt)
#RHS_u9_sep = 0.5*(R_cath + R_cath_prevt)
RHS_u9_sep = R_cath
u9_sep = LHS_u9_sep - RHS_u9_sep

# =============================================================================
# ## u10_sep for e_sep ## 
# =============================================================================
LHS_u10_sep = (e_sep - e_sep_prevt)/(dt)
#RHS_u10_sep = 0.5*(R_sep + R_sep_prevt)
RHS_u10_sep = R_sep
u10_sep = LHS_u10_sep - RHS_u10_sep

# =============================================================================
# ## u11_sep for e_s8_cath ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u11_sep = (e_s8_cath - e_s8_cath_prevt)/(dt)
RHS_u11_sep = V_s8*k_s8*e_s8_cath*(s8 - ksp_s8)
u11_sep = LHS_u11_sep - RHS_u11_sep

# =============================================================================
# ## u12_sep for e_s8_sep ## 
# =============================================================================
LHS_u12_sep = (e_s8_sep - e_s8_sep_prevt)/(dt)
RHS_u12_sep = V_s8*k_s8*e_s8_sep*(s8 - ksp_s8)
u12_sep = LHS_u12_sep - RHS_u12_sep

# =============================================================================
# ## u13_sep for e_Li2s_cath ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u13_sep = (e_Li2s_cath - e_Li2s_cath_prevt)/(dt)
RHS_u13_sep = V_Li2s*k_Li2s*e_Li2s_cath*((Li**2)*(s1) - ksp_Li2s)
u13_sep = LHS_u13_sep - RHS_u13_sep

# =============================================================================
# ## u14_sep for e_Li2s_sep ## 
# =============================================================================
LHS_u14_sep = (e_Li2s_sep - e_Li2s_sep_prevt)/(dt)
RHS_u14_sep = V_Li2s*k_Li2s*e_Li2s_sep*((Li**2)*(s1) - ksp_Li2s)
u14_sep = LHS_u14_sep - RHS_u14_sep

# =============================================================================
# ## u15_sep for phi1 at separator ## (This will be removed via Penalty Method)
# =============================================================================
LHS_u15_sep = (z_Li*div_N_Li_sep + z_s8*div_N_s8_sep + z_s8_2*div_N_s8_2_sep + z_s6*div_N_s6_sep + z_s4*div_N_s4_sep + z_s2*div_N_s2_sep + z_s1*div_N_s1_sep + z_An*div_N_An_sep)
RHS_u15_sep = 0*(phi1_postx - 2*phi1 + phi1_prevx)/(dx**2) # sigma = 0 in separator
u15_sep = LHS_u15_sep - RHS_u15_sep

# =============================================================================
# ## u16_sep for phi2 at separator ##
# =============================================================================
LHS_u16_sep = LHS_u15_sep
RHS_u16_sep = 0*(i2 + i3 + i4 + i5 + i6) ## No partial currents in separator
u16_sep = LHS_u16_sep - RHS_u16_sep

## Overall u-function-arrays
u_list_cath = [u1_cath, u2_cath, u3_cath, u4_cath, u5_cath, u6_cath, u7_cath, u8_cath, u9_cath, u10_cath, u11_cath, u12_cath, u13_cath, u14_cath, u15_cath, u16_cath]
u_list_sep = [u1_sep, u2_sep, u3_sep, u4_sep, u5_sep, u6_sep, u7_sep, u8_sep, u9_sep, u10_sep, u11_sep, u12_sep, u13_sep, u14_sep, u15_sep, u16_sep]

variables = [Li, s8, s8_2, s6, s4, s2, s1, An, e_cath, e_sep, e_s8_cath, e_s8_sep, e_Li2s_cath, e_Li2s_sep, phi1, phi2]

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
# Remove variable dependancies in separator & cathode region (Via Penalty Method)
# =============================================================================
## Remove phi1 from separator
index_separator_points_phi1 = []
for i in range(1, index_sep_cath_boundary):
    index_separator_points_phi1.append(i)

## Remove e_cath from separator
index_separator_points_e_cath = []
for i in range(0, index_sep_cath_boundary): # Test bleeding method (Allow e_p 1 point into region of non existence)
    index_separator_points_e_cath.append(i)

## Remove e_s8_cath from separator
index_separator_points_e_s8_cath = index_separator_points_e_cath.copy()

## Remove e_Li2s_cath from separator
index_separator_points_e_Li2s_cath = index_separator_points_e_cath.copy()

## Remove e_sep from cathode
## Test with including both e_cath & e_sep at the cath/sep interface boundary point
index_cathode_points_e_sep = []
for i in range(index_sep_cath_boundary + 1, num_space): ## Test with no e_sep at sep/cath boundary OR Test bleeding method (Allow e_p 1 point into region of non existence)
    index_cathode_points_e_sep.append(i)
    
## Remove e_s8_sep from cathode
index_cathode_points_e_s8_sep = index_cathode_points_e_sep.copy()

## Remove e_Li2s_sep from cathode
index_cathode_points_e_Li2s_sep = index_cathode_points_e_sep.copy()

## Appened row indices for Penalty Method
index_to_delete = []
for i in range(num_rows):
    ## phi1
    if row_label[i][0] == phi1 and row_label[i][1] in index_separator_points_phi1:
        index_to_delete.append(i)
    ## e_cath
    if row_label[i][0] == e_cath and row_label[i][1] in index_separator_points_e_cath:
        index_to_delete.append(i)
    ## e_s8_cath
    if row_label[i][0] == e_s8_cath and row_label[i][1] in index_separator_points_e_s8_cath:
        index_to_delete.append(i)
    ## e_Li2s_cath
    if row_label[i][0] == e_Li2s_cath and row_label[i][1] in index_separator_points_e_Li2s_cath:
        index_to_delete.append(i)
    ## e_sep
    if row_label[i][0] == e_sep and row_label[i][1] in index_cathode_points_e_sep:
        index_to_delete.append(i)
    ## e_s8_sep
    if row_label[i][0] == e_s8_sep and row_label[i][1] in index_cathode_points_e_s8_sep:
        index_to_delete.append(i)
    ## e_Li2s_sep
    if row_label[i][0] == e_Li2s_sep and row_label[i][1] in index_cathode_points_e_Li2s_sep:
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

## Penalty Method after the boundary conditions are filled
# =============================================================================
# ## Introduce the penalty method
# for i in range(num_rows):
#     for j in range(num_cols):
#         if i == j and i in index_to_delete:
#             jacob[i][j] = jacob[i][j] + 1e50 ## Add a large penalty factor
# =============================================================================
            
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
## Species flux equations at boundary (defined wrt to separator constants *D_i_sep)
N_Li_sep_x0 = -D_Li_sep*(Li_postx - Li)/(dx) - ((z_Li*D_Li_sep*F)/(R*T))*(Li)*(phi2_postx - phi2)/(dx)
N_s8_sep_x0 = -D_s8_sep*(s8_postx - s8)/(dx) - ((z_s8*D_s8_sep*F)/(R*T))*(s8)*(phi2_postx - phi2)/(dx)
N_s8_2_sep_x0 = -D_s8_2_sep*(s8_2_postx - s8_2)/(dx) - ((z_s8_2*D_s8_2_sep*F)/(R*T))*(s8_2)*(phi2_postx - phi2)/(dx)
N_s6_sep_x0 = -D_s6_sep*(s6_postx - s6)/(dx) - ((z_s6*D_s6_sep*F)/(R*T))*(s6)*(phi2_postx - phi2)/(dx)
N_s4_sep_x0 = -D_s4_sep*(s4_postx - s4)/(dx) - ((z_s4*D_s4_sep*F)/(R*T))*(s4)*(phi2_postx - phi2)/(dx)
N_s2_sep_x0 = -D_s2_sep*(s2_postx - s2)/(dx) - ((z_s2*D_s2_sep*F)/(R*T))*(s2)*(phi2_postx - phi2)/(dx)
N_s1_sep_x0 = -D_s1_sep*(s1_postx - s1)/(dx) - ((z_s1*D_s1_sep*F)/(R*T))*(s1)*(phi2_postx - phi2)/(dx)
N_An_sep_x0 = -D_An_sep*(An_postx - An)/(dx) - ((z_An*D_An_sep*F)/(R*T))*(An)*(phi2_postx - phi2)/(dx)
flux_sum_sep_x0 = (z_Li*N_Li_sep_x0 + z_s8*N_s8_sep_x0 + z_s8_2*N_s8_2_sep_x0 + z_s6*N_s6_sep_x0 + z_s4*N_s4_sep_x0 + z_s2*N_s2_sep_x0 + z_s1*N_s1_sep_x0 + z_An*N_An_sep_x0)

## Apply boundary conditions at x=0
ub1_x0 = F*N_Li_sep_x0 - i1
ub2_x0 = N_s8_sep_x0
ub3_x0 = N_s8_2_sep_x0
ub4_x0 = N_s6_sep_x0
ub5_x0 = N_s4_sep_x0
ub6_x0 = N_s2_sep_x0
ub7_x0 = N_s1_sep_x0
ub8_x0 = N_An_sep_x0
## The porosity variables do not have boundary conditions, same equations used from above
ub9_x0 = LHS_u9_sep - RHS_u9_sep
ub10_x0 = LHS_u10_sep - RHS_u10_sep
ub11_x0 = LHS_u11_sep - RHS_u11_sep
ub12_x0 = LHS_u12_sep - RHS_u12_sep
ub13_x0 = LHS_u13_sep - RHS_u13_sep
ub14_x0 = LHS_u14_sep - RHS_u14_sep
## phi1 and phi2 boundary conditions
ub15_x0 = phi1 # phi1 = 0 at x = 0
#ub16_x0 = i1 - (I_app/A)
ub16_x0 = phi2_postx - phi2
#ub16_x0 = F*N_Li_sep_x0 - F*flux_sum_sep_x0

ub_x0_list = [ub1_x0, ub2_x0, ub3_x0, ub4_x0, ub5_x0, ub6_x0, ub7_x0, ub8_x0, ub9_x0, ub10_x0, ub11_x0, ub12_x0, ub13_x0, ub14_x0, ub15_x0, ub16_x0]

# =============================================================================
# At x=Ls (Seperator/Cathode interface) boundary conditions 
# =============================================================================
### Test 1 ###
# =============================================================================
# ## Dynamic coefficients at cathode side of interface:
# D_Li_cath1 = D_Li*(e_cath_postx**1.5)
# D_s8_cath1 = D_s8*(e_cath_postx**1.5)
# D_s8_2_cath1 = D_s8_2*(e_cath_postx**1.5)
# D_s6_cath1 = D_s6*(e_cath_postx**1.5)
# D_s4_cath1 = D_s4*(e_cath_postx**1.5)
# D_s2_cath1 = D_s2*(e_cath_postx**1.5)
# D_s1_cath1 = D_s1*(e_cath_postx**1.5)
# D_An_cath1 = D_An*(e_cath_postx**1.5)
# ## Dynamic coefficients at seperator side of interface:
# D_Li_sep1 = D_Li*(e_sep_prevx**1.5)
# D_s8_sep1 = D_s8*(e_sep_prevx**1.5)
# D_s8_2_sep1 = D_s8_2*(e_sep_prevx**1.5)
# D_s6_sep1 = D_s6*(e_sep_prevx**1.5)
# D_s4_sep1 = D_s4*(e_sep_prevx**1.5)
# D_s2_sep1 = D_s2*(e_sep_prevx**1.5)
# D_s1_sep1 = D_s1*(e_sep_prevx**1.5)
# D_An_sep1 = D_An*(e_sep_prevx**1.5)
# =============================================================================

### Test 2 ###
## Dynamic coefficients at cathode side of interface:
D_Li_cath1 = D_Li*(e_cath**1.5)
D_s8_cath1 = D_s8*(e_cath**1.5)
D_s8_2_cath1 = D_s8_2*(e_cath**1.5)
D_s6_cath1 = D_s6*(e_cath**1.5)
D_s4_cath1 = D_s4*(e_cath**1.5)
D_s2_cath1 = D_s2*(e_cath**1.5)
D_s1_cath1 = D_s1*(e_cath**1.5)
D_An_cath1 = D_An*(e_cath**1.5)
## Dynamic coefficients at seperator side of interface:
D_Li_sep1 = D_Li*(e_sep**1.5)
D_s8_sep1 = D_s8*(e_sep**1.5)
D_s8_2_sep1 = D_s8_2*(e_sep**1.5)
D_s6_sep1 = D_s6*(e_sep**1.5)
D_s4_sep1 = D_s4*(e_sep**1.5)
D_s2_sep1 = D_s2*(e_sep**1.5)
D_s1_sep1 = D_s1*(e_sep**1.5)
D_An_sep1 = D_An*(e_sep**1.5)

## Flux sum at seperator section
N_Li_sep_xs = -D_Li_sep1*(Li - Li_prevx)/(dx) - ((z_Li*D_Li_sep1*F)/(R*T))*(Li)*(phi2 - phi2_prevx)/(dx)
N_s8_sep_xs = -D_s8_sep1*(s8 - s8_prevx)/(dx) - ((z_s8*D_s8_sep1*F)/(R*T))*(s8)*(phi2 - phi2_prevx)/(dx)
N_s8_2_sep_xs = -D_s8_2_sep1*(s8_2 - s8_2_prevx)/(dx) - ((z_s8_2*D_s8_2_sep1*F)/(R*T))*(s8_2)*(phi2 - phi2_prevx)/(dx)
N_s6_sep_xs = -D_s6_sep1*(s6 - s6_prevx)/(dx) - ((z_s6*D_s6_sep1*F)/(R*T))*(s6)*(phi2 - phi2_prevx)/(dx)
N_s4_sep_xs = -D_s4_sep1*(s4 - s4_prevx)/(dx) - ((z_s4*D_s4_sep1*F)/(R*T))*(s4)*(phi2 - phi2_prevx)/(dx)
N_s2_sep_xs = -D_s2_sep1*(s2 - s2_prevx)/(dx) - ((z_s2*D_s2_sep1*F)/(R*T))*(s2)*(phi2 - phi2_prevx)/(dx)
N_s1_sep_xs = -D_s1_sep1*(s1 - s1_prevx)/(dx) - ((z_s1*D_s1_sep1*F)/(R*T))*(s1)*(phi2 - phi2_prevx)/(dx)
N_An_sep_xs = -D_An_sep1*(An - An_prevx)/(dx) - ((z_An*D_An_sep1*F)/(R*T))*(An)*(phi2 - phi2_prevx)/(dx)
flux_sum_sep_xs = (z_Li*N_Li_sep_xs + z_s8*N_s8_sep_xs + z_s8_2*N_s8_2_sep_xs + z_s6*N_s6_sep_xs + z_s4*N_s4_sep_xs + z_s2*N_s2_sep_xs + z_s1*N_s1_sep_xs + z_An*N_An_sep_xs)
## Flux sum at cathode section
N_Li_cath_xs = -D_Li_cath1*(Li_postx - Li)/(dx) - ((z_Li*D_Li_cath1*F)/(R*T))*(Li)*(phi2_postx - phi2)/(dx)
N_s8_cath_xs = -D_s8_cath1*(s8_postx - s8)/(dx) - ((z_s8*D_s8_cath1*F)/(R*T))*(s8)*(phi2_postx - phi2)/(dx)
N_s8_2_cath_xs = -D_s8_2_cath1*(s8_2_postx - s8_2)/(dx) - ((z_s8_2*D_s8_2_cath1*F)/(R*T))*(s8_2)*(phi2_postx - phi2)/(dx)
N_s6_cath_xs = -D_s6_cath1*(s6_postx - s6)/(dx) - ((z_s6*D_s6_cath1*F)/(R*T))*(s6)*(phi2_postx - phi2)/(dx)
N_s4_cath_xs = -D_s4_cath1*(s4_postx - s4)/(dx) - ((z_s4*D_s4_cath1*F)/(R*T))*(s4)*(phi2_postx - phi2)/(dx)
N_s2_cath_xs = -D_s2_cath1*(s2_postx - s2)/(dx) - ((z_s2*D_s2_cath1*F)/(R*T))*(s2)*(phi2_postx - phi2)/(dx)
N_s1_cath_xs = -D_s1_cath1*(s1_postx - s1)/(dx) - ((z_s1*D_s1_cath1*F)/(R*T))*(s1)*(phi2_postx - phi2)/(dx)
N_An_cath_xs = -D_An_cath1*(An_postx - An)/(dx) - ((z_An*D_An_cath1*F)/(R*T))*(An)*(phi2_postx - phi2)/(dx)
flux_sum_cath_xs = (z_Li*N_Li_cath_xs + z_s8*N_s8_cath_xs + z_s8_2*N_s8_2_cath_xs + z_s6*N_s6_cath_xs + z_s4*N_s4_cath_xs + z_s2*N_s2_cath_xs + z_s1*N_s1_cath_xs + z_An*N_An_cath_xs)

## Apply boundary conditions at x = Ls
ub1_xs = N_Li_cath_xs - N_Li_sep_xs
ub2_xs = N_s8_cath_xs - N_s8_sep_xs
ub3_xs = N_s8_2_cath_xs - N_s8_2_sep_xs
ub4_xs = N_s6_cath_xs - N_s6_sep_xs
ub5_xs = N_s4_cath_xs - N_s4_sep_xs
ub6_xs = N_s2_cath_xs - N_s2_sep_xs
ub7_xs = N_s1_cath_xs - N_s1_sep_xs
ub8_xs = N_An_cath_xs - N_An_sep_xs
## The porosity variables do not have boundary conditions, same equations used from above
ub9_xs = LHS_u9_sep - RHS_u9_sep
ub10_xs = LHS_u10_sep - RHS_u10_sep
ub11_xs = LHS_u11_sep - RHS_u11_sep
ub12_xs = LHS_u12_sep - RHS_u12_sep
ub13_xs = LHS_u13_sep - RHS_u13_sep
ub14_xs = LHS_u14_sep - RHS_u14_sep
## phi1 & phi2 boundary conditions
ub15_xs = (phi1_postx - phi1)/(dx)
#ub16_xs = F*flux_sum_sep_xs + F*flux_sum_cath_xs - 2*(I_app/(A))
#ub16_xs = flux_sum_sep_xs + flux_sum_cath_xs - 2*(I_app/(A*F))
#ub16_xs = F*flux_sum_sep_xs - (I_app/(A))
#ub16_xs = flux_sum_sep_xs - (I_app/(F*A))
ub16_xs = flux_sum_sep_xs - flux_sum_cath_xs
## Try the below:
#ub16_xs = (F*flux_sum_sep_xs - F*flux_sum_cath_xs) + (F*flux_sum_sep_xs - (I_app/(A))) + (F*flux_sum_cath_xs - (I_app/(A)))
#ub16_xs = (F*flux_sum_sep_xs - F*flux_sum_cath_xs)**2 + (F*flux_sum_sep_xs - (I_app/(A)))**2 + (F*flux_sum_cath_xs - (I_app/(A)))**2

ub_xs_list = [ub1_xs, ub2_xs, ub3_xs, ub4_xs, ub5_xs, ub6_xs, ub7_xs, ub8_xs, ub9_xs, ub10_xs, ub11_xs, ub12_xs, ub13_xs, ub14_xs, ub15_xs, ub16_xs]

# =============================================================================
# At x=Ls+Lc (Cathode current collector) boundary conditions 
# =============================================================================
N_Li_cath_xL = -D_Li_cath*(Li - Li_prevx)/(dx) - ((z_Li*D_Li_cath*F)/(R*T))*(Li)*(phi2 - phi2_prevx)/(dx)
N_s8_cath_xL = -D_s8_cath*(s8 - s8_prevx)/(dx) - ((z_s8*D_s8_cath*F)/(R*T))*(s8)*(phi2 - phi2_prevx)/(dx)
N_s8_2_cath_xL = -D_s8_2_cath*(s8_2 - s8_2_prevx)/(dx) - ((z_s8_2*D_s8_2_cath*F)/(R*T))*(s8_2)*(phi2 - phi2_prevx)/(dx)
N_s6_cath_xL = -D_s6_cath*(s6 - s6_prevx)/(dx) - ((z_s6*D_s6_cath*F)/(R*T))*(s6)*(phi2 - phi2_prevx)/(dx)
N_s4_cath_xL = -D_s4_cath*(s4 - s4_prevx)/(dx) - ((z_s4*D_s4_cath*F)/(R*T))*(s4)*(phi2 - phi2_prevx)/(dx)
N_s2_cath_xL = -D_s2_cath*(s2 - s2_prevx)/(dx) - ((z_s2*D_s2_cath*F)/(R*T))*(s2)*(phi2 - phi2_prevx)/(dx)
N_s1_cath_xL = -D_s1_cath*(s1 - s1_prevx)/(dx) - ((z_s1*D_s1_cath*F)/(R*T))*(s1)*(phi2 - phi2_prevx)/(dx)
N_An_cath_xL = -D_An_cath*(An - An_prevx)/(dx) - ((z_An*D_An_cath*F)/(R*T))*(An)*(phi2 - phi2_prevx)/(dx)
flux_sum_cath_xL = (z_Li*N_Li_cath_xL + z_s8*N_s8_cath_xL + z_s8_2*N_s8_2_cath_xL + z_s6*N_s6_cath_xL + z_s4*N_s4_cath_xL + z_s2*N_s2_cath_xL + z_s1*N_s1_cath_xL + z_An*N_An_cath_xL)

## Apply boundary conditions at x = Ls+Lc
ub1_xL = Li - Li_prevx
ub2_xL = s8 - s8_prevx
ub3_xL = s8_2 - s8_2_prevx
ub4_xL = s6 - s6_prevx
ub5_xL = s4 - s4_prevx
ub6_xL = s2 - s2_prevx
ub7_xL = s1 - s1_prevx
ub8_xL = An - An_prevx
## The porosity variables do not have boundary conditions, same equations used from above
ub9_xL = LHS_u9_sep - RHS_u9_sep
ub10_xL = LHS_u10_sep - RHS_u10_sep
ub11_xL = LHS_u11_sep - RHS_u11_sep
ub12_xL = LHS_u12_sep - RHS_u12_sep
ub13_xL = LHS_u13_sep - RHS_u13_sep
ub14_xL = LHS_u14_sep - RHS_u14_sep
## phi1 & phi2 boundary conditions
#ub15_xL = phi1 - phi1_prevx + ((I_app*dx)/(sigma*A))
ub15_xL = sigma*(phi1 - phi1_prevx)/(dx) + ((I_app)/(A))
ub16_xL = phi2 - phi2_prevx

ub_xL_list = [ub1_xL, ub2_xL, ub3_xL, ub4_xL, ub5_xL, ub6_xL, ub7_xL, ub8_xL, ub9_xL, ub10_xL, ub11_xL, ub12_xL, ub13_xL, ub14_xL, ub15_xL, ub16_xL]

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

## Introduce the penalty method now
for i in range(num_rows):
    for j in range(num_cols):
        if i == j and i in index_to_delete:
            jacob[i][j] = jacob[i][j] + 1e5 ## Add a large penalty factor

# =============================================================================
# Check Matrix Sparsity         
# =============================================================================
to_check = [element for sublist in jacob for element in sublist]
no_zeros = 0
for i in range(len(to_check)):
    if to_check[i] == 0:
        no_zeros += 1
        
percentage_sparsity = 100*(no_zeros/len(to_check))
print('% Sparsity of Jacobian Matrix: ', percentage_sparsity)

# =============================================================================
# Lambdify both the u_func_array and jacob
# =============================================================================
## For loop to lambdify the Jacobian functions
print('Lambdifying Jacobian Matrix:')
count_jacob = 0
for i in range(num_rows):
    for j in range(num_cols):
        if jacob[i][j] != 0:
            jacob[i][j] = lambdify(syms, jacob[i][j], 'numpy')
            count_jacob += 1
            update_loading_bar(count_jacob, ((num_var*num_space)**2 - no_zeros))

print()
print('Jacobian Matrix Lambdification Complete !!!')

## For loop to lambdify the array functions
print('Lambdifying Function Arrays:')
count_u = 0
for i in range(len(u_func_array)):
    u_func_array[i] = lambdify(syms, u_func_array[i], 'numpy')
    count_u += 1
    update_loading_bar(count_u, num_var*num_space)

print()
print('Function Arrays Lambdification Complete !!!')  

## Now we make a model class to carry out substitutions and solver function to solve ##
## These will be done in another script for ease of sub-dividing scripts ##