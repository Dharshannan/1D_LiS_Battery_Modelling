import numpy as np 
import matplotlib.pyplot as plt
import func_potentials as func
# =============================================================================
# Model parameters
# =============================================================================
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
k_s8 = 5 #1
k_Li2s = 6.9e-5 #3.45e-5 #100 #27.5 
ksp_s8 = 19
ksp_Li2s = 9.95e3 #9.95e-4 * 4e8 #9.95e3 #1e2 #3e-5 #1.55e7            
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
I_app = 0.5 #0.394*A ## (NOTE: '-' for charge '+' for discharge)
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

## Load data 
data = np.load('var_array_0.5A_constant.npz', allow_pickle=True)
overall_array = data['solved'][:, :-2]
time = data['time'][:-2]
x = func.x_space
I = abs(I_app)
A = func.A
dx = x[1] - x[0]

# =============================================================================
# Calculate partial currents integral in cathode
# =============================================================================
bd = 2 ## sep/cath interface cut-off point (Minimize instabilities)
tm = 3 ## initial time cut-off (Minimize instabilities)
## Load required potentials and concentrations across the cathode
s8 = overall_array[1, :, func.index_sep_cath_boundary+bd:]
s8_2 = overall_array[2, :, func.index_sep_cath_boundary+bd:]
s6 = overall_array[3, :, func.index_sep_cath_boundary+bd:]
s4 = overall_array[4, :, func.index_sep_cath_boundary+bd:]
s2 = overall_array[5, :, func.index_sep_cath_boundary+bd:]
s1 = overall_array[6, :, func.index_sep_cath_boundary+bd:]
phi1 = overall_array[-2, :, func.index_sep_cath_boundary+bd:]
phi2 = overall_array[-1, :, func.index_sep_cath_boundary+bd:]
e_cath = overall_array[8, :, func.index_sep_cath_boundary+bd:]

## Define U_ref
U1_ref = U1
U2_ref = U2 - ((R*T)/(2*F))*np.log(C_s8_2/C_s8)
U3_ref = U3 - ((R*T)/(F))*(-1.5*(np.log(C_s8_2/1e3)) + 2*(np.log(C_s6/1e3)))
U4_ref = U4 - ((R*T)/(F))*(-1*np.log(C_s6/1e3) + 1.5*np.log(C_s4/1e3))
U5_ref = U5 - ((R*T)/(F))*(-0.5*np.log(C_s4/1e3) + np.log(C_s2/1e3))
U6_ref = U6 - ((R*T)/(F))*(-0.5*np.log(C_s2/1e3) + np.log(C_s1/1e3))
## calc partial currents ##
## i2
i2 = []
for i in range(len(s8)):
    eta2 = phi1[i] - phi2[i] - U2_ref
    partial = i02*(((s8_2[i]/C_s8_2)**0.5)*np.exp(0.5*(F/(R*T))*eta2) - ((s8[i]/C_s8)**0.5)*np.exp(-0.5*(F/(R*T))*eta2))
    av = a0*((e_cath[i]/e_cath_init)**1.5)
    current_density = np.sum(av*partial*dx)
    normalized = (1/(I/A))*current_density
    normalized = normalized*-1
    i2.append(normalized)
    
i2 = np.asarray(i2)
plt.plot(I*time[tm:]/3600, i2[tm:], label='${I_2}^{N}$')

## i3
i3 = []
for i in range(len(s8)):
    eta3 = phi1[i] - phi2[i] - U3_ref
    partial = i03*(((s6[i]/C_s6)**2)*np.exp(0.5*(F/(R*T))*eta3) - ((s8_2[i]/C_s8_2)**1.5)*np.exp(-0.5*(F/(R*T))*eta3))
    av = a0*((e_cath[i]/e_cath_init)**1.5)
    current_density = np.sum(av*partial*dx)
    normalized = (1/(I/A))*current_density
    normalized = normalized*-1
    i3.append(normalized)
    
i3 = np.asarray(i3)
plt.plot(I*time[tm:]/3600, i3[tm:], label='${I_3}^{N}$')

## i4
i4 = []
for i in range(len(s8)):
    eta4 = phi1[i] - phi2[i] - U4_ref
    partial = i04*(((s4[i]/C_s4)**1.5)*np.exp(0.5*(F/(R*T))*eta4) - (s6[i]/C_s6)*np.exp(-0.5*(F/(R*T))*eta4))
    av = a0*((e_cath[i]/e_cath_init)**1.5)
    current_density = np.sum(av*partial*dx)
    normalized = (1/(I/A))*current_density
    normalized = normalized*-1
    i4.append(normalized)
    
i4 = np.asarray(i4)
plt.plot(I*time[tm:]/3600, i4[tm:], label='${I_4}^{N}$')

## i5
i5 = []
for i in range(len(s8)):
    eta5 = phi1[i] - phi2[i] - U5_ref
    partial = i05*((s2[i]/C_s2)*np.exp(0.5*(F/(R*T))*eta5) - ((s4[i]/C_s4)**0.5)*np.exp(-0.5*(F/(R*T))*eta5))
    av = a0*((e_cath[i]/e_cath_init)**1.5)
    current_density = np.sum(av*partial*dx)
    normalized = (1/(I/A))*current_density
    normalized = normalized*-1
    i5.append(normalized)
    
i5 = np.asarray(i5)
plt.plot(I*time[tm:]/3600, i5[tm:], label='${I_5}^{N}$')

## i6
i6 = []
for i in range(len(s8)):
    eta6 = phi1[i] - phi2[i] - U6_ref
    partial = i06*((s1[i]/C_s1)*np.exp(0.5*(F/(R*T))*eta6) - ((s2[i]/C_s2)**0.5)*np.exp(-0.5*(F/(R*T))*eta6))
    av = a0*((e_cath[i]/e_cath_init)**1.5)
    current_density = np.sum(av*partial*dx)
    normalized = (1/(I/A))*current_density
    normalized = normalized*-1
    i6.append(normalized)
    
i6 = np.asarray(i6)
plt.plot(I*time[tm:]/3600, i6[tm:], label='${I_6}^{N}$')

## Take sum of the currents
currents_sum = i2 + i3 + i4 + i5 + i6
plt.plot(I*time[tm:]/3600, currents_sum[tm:], label='${I_{Total}}^{N}$', linestyle='dashed')
plt.ylabel('${I_j}^{N}$')
plt.xlabel('Discharge Capacity [Ah]')
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
#plt.ylim([-2,2])
plt.title('Voltage Cycle Current: 0.25A (Sine)')
#plt.savefig('Partial_Currents_Sine.png', dpi=1500, bbox_inches='tight')
plt.show()

# =============================================================================
# Calculate charge balance
# =============================================================================
s_array = overall_array[0:8, :] ## List of all species
z = [z_Li, z_s8, z_s8_2, z_s6, z_s4, z_s2, z_s1, z_An] ## Species charges
e_cath = overall_array[8] ## Cathode porosity as above
e_sep = overall_array[9] ## Separator porosity

## a) Charge at every spatial points over time
for i in range(0, len(s_array[0]), int(s_array.shape[1]/10)):
    spatial_charge = []
    for j in range(s_array.shape[2]):
        if j == 0:
            point_charge = 0.5*s_array[:, i, j]*z*e_sep[i, j]
            spatial_charge.append(np.sum(point_charge))
        
        if j == func.index_sep_cath_boundary:
            point_charge = 0.5*(s_array[:, i, j]*z*e_sep[i, j] + s_array[:, i, j]*z*e_cath[i, j])
            spatial_charge.append(np.sum(point_charge))
            
        if j == s_array.shape[2] - 1:
            point_charge = 0.5*s_array[:, i, j]*z*e_cath[i, j]
            spatial_charge.append(np.sum(point_charge))
            
        else:
            if j>0 and j<func.index_sep_cath_boundary:
                point_charge = s_array[:, i, j]*z*e_sep[i, j]
                spatial_charge.append(np.sum(point_charge))
            
            if j > func.index_sep_cath_boundary and j < s_array.shape[2] - 1:
                point_charge = s_array[:, i, j]*z*e_cath[i, j]
                spatial_charge.append(np.sum(point_charge))
                
    spatial_charge = np.asarray(spatial_charge)
    plt.plot(x, spatial_charge, label=f'time: {round(time[i], 0)}')

plt.ylabel('Total Charge [+ -]')
plt.xlabel('Cell Distance [m]')
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
#plt.savefig('Spatial_Charge.png', dpi=500, bbox_inches='tight')
plt.show()

## b) Average charge of whole cell overtime
avg_charge = []
for i in range(0, len(s_array[0])):
    spatial_charge = []
    for j in range(s_array.shape[2]):
        if j == 0:
            point_charge = 0.5*s_array[:, i, j]*z*e_sep[i, j]
            spatial_charge.append(np.sum(point_charge))
        
        if j == func.index_sep_cath_boundary:
            point_charge = 0.5*(s_array[:, i, j]*z*e_sep[i, j] + s_array[:, i, j]*z*e_cath[i, j])
            spatial_charge.append(np.sum(point_charge))
            
        if j == s_array.shape[2] - 1:
            point_charge = 0.5*s_array[:, i, j]*z*e_cath[i, j]
            spatial_charge.append(np.sum(point_charge))
            
        else:
            if j>0 and j<func.index_sep_cath_boundary:
                point_charge = s_array[:, i, j]*z*e_sep[i, j]
                spatial_charge.append(np.sum(point_charge))
            
            if j > func.index_sep_cath_boundary and j < s_array.shape[2] - 1:
                point_charge = s_array[:, i, j]*z*e_cath[i, j]
                spatial_charge.append(np.sum(point_charge))
    
    spatial_charge = np.asarray(spatial_charge)
    avg = np.sum(spatial_charge)/(s_array.shape[2])
    avg_charge.append(avg)
    
avg_charge = np.asarray(avg_charge)
#plt.plot(I*time[tm:]/3600, avg_charge[tm:])

## Plot error
charge_error = []
for i in range(len(avg_charge)):
    err = abs(avg_charge[i] - 0)*100
    charge_error.append(err)
    
#plt.plot(I*time[tm:]/3600, charge_error[tm:])

## Plot both on same axes
fig, ax1 = plt.subplots()
#ax1.plot(time[1:]/3600, h, c='steelblue')
ax1.plot(I*time[tm:]/3600, avg_charge[tm:], c='steelblue')
ax1.set_ylabel('Average Cell Charge [+ -]', c='steelblue')
ax1.set_xlabel('Discharge Capacity [Ah]')

ax2 = ax1.twinx()
ax2.plot(I*time[tm:]/3600, charge_error[tm:], c='darkorange', linestyle='dashed')
ax2.set_ylabel('Error [%]', c='darkorange')
ax2.set_xlabel('Discharge Capacity [Ah]')

ax1.spines['left'].set_color('steelblue')     # Left spine
ax2.spines['right'].set_color('darkorange')     # Right spine
ax1.yaxis.label.set_color('steelblue')        # Y1-axis label
ax2.yaxis.label.set_color('darkorange')         # Y2-axis label
ax1.tick_params(axis='y', colors='steelblue')  # Y1-axis ticks
ax2.tick_params(axis='y', colors='darkorange')   # Y2-axis ticks
plt.title('Voltage Cycle Current: 0.25A')
#plt.savefig('Average_Cell_Charge_Sine.png', dpi=1500, bbox_inches='tight')
plt.show()