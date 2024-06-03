import numpy as np 
import matplotlib.pyplot as plt
import func_potentials as func
from scipy.signal import butter, filtfilt

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
## Low Pass filter function to filter noise from data    
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

data = np.load('var_array_0.5A_constant.npz', allow_pickle=True)
overall_array = data['solved'][:, :-1]
time = data['time'][:-1]
x = func.x_space
I = 0.5
dx = x[1] - x[0]

var = ['$Li^+$', '$s_{8(l)}$', '${s_8}^{2-}$', '${s_6}^{2-}$', '${s_4}^{2-}$', '${s_2}^{2-}$', '$s^{2-}$', '$A^{-}$', '$\phi_1$', '$\phi_2$']
var2 = ['Li+', 's8l', 's8_2-', 's6_2-', 's4_2-', 's2_2-', 's_2-', 'A-', 'phi1', 'phi2']

# =============================================================================
# ## Calculate Anion Salt Mass Conservation
# =============================================================================
An_salt = overall_array[7, :]
e_cath = overall_array[8]
e_sep = overall_array[9]
An_mass = np.zeros(len(An_salt))

for i in range(len(An_salt)):
    #An_mass[i] = np.sum(An_salt[i])
    An_anode = 0.5*e_sep[i][0]*An_salt[i][0]
    An_sep = np.sum(e_sep[i][1:func.index_sep_cath_boundary]*An_salt[i][1:func.index_sep_cath_boundary])
    An_sep_int = 0.5*e_sep[i][func.index_sep_cath_boundary]*An_salt[i][func.index_sep_cath_boundary]
    An_cath_int = 0.5*e_cath[i][func.index_sep_cath_boundary]*An_salt[i][func.index_sep_cath_boundary]
    An_cath = np.sum(e_cath[i][func.index_sep_cath_boundary:-1]*An_salt[i][func.index_sep_cath_boundary:-1])
    An_cath_bd = 0.5*e_cath[i][-1]*An_salt[i][-1]
    An_mass[i] = (An_anode + An_sep + An_sep_int + An_cath_int + An_cath + An_cath_bd)*(1/(func.L_cath + func.L_sep))*dx

plt.plot(time/3600, An_mass)
plt.ylabel(f'Total {var[-3]} Concentration [$mol/m^3$]')
plt.xlabel('Time [h]')
#plt.savefig('Anion_Salt.png', dpi=500, bbox_inches='tight')
plt.show()

## Calculate and plot error wrt Anion mass conservation
mass_error = []
for i in range(1, len(An_salt)):
    err = (abs(An_mass[i] - An_mass[0])/An_mass[0])*100
    mass_error.append(err)
mass_error = np.asarray(mass_error)
plt.plot(time[1:]/3600, mass_error)
plt.ylabel('Error [%]')
plt.xlabel('Time [h]')
#plt.savefig('Anion_Salt.png', dpi=500, bbox_inches='tight')
plt.show()

## Plot both on same axes
fig, ax1 = plt.subplots()
#ax1.plot(time[1:]/3600, h, c='steelblue')
ax1.plot(abs(I)*time/3600, An_mass, c='steelblue')
ax1.set_ylabel(f'Total {var[-3]} Concentration [$mol/m^3$]', c='steelblue')
ax1.set_xlabel('Discharge Capacity [Ah]')

ax2 = ax1.twinx()
ax2.plot(abs(I)*time[1:]/3600, mass_error, c='darkorange', linestyle='dashed')
ax2.set_ylabel('Error [%]', c='darkorange')
ax2.set_xlabel('Discharge Capacity [Ah]')

ax1.spines['left'].set_color('steelblue')     # Left spine
ax2.spines['right'].set_color('darkorange')     # Right spine
ax1.yaxis.label.set_color('steelblue')        # Y1-axis label
ax2.yaxis.label.set_color('darkorange')         # Y2-axis label
ax1.tick_params(axis='y', colors='steelblue')  # Y1-axis ticks
ax2.tick_params(axis='y', colors='darkorange')   # Y2-axis ticks
plt.title('Voltage Cycle Current: 0.25A (Sine)')
#plt.savefig('Anion_Salt_Mass_Error_Sine.png', dpi=1500, bbox_inches='tight')
plt.show()

## Calculate Lithium Mass Conservation
Li = overall_array[0, :]
Li_mass = np.zeros(len(Li))

for i in range(len(Li)):
    Li_mass[i] = np.sum(Li[i])*(1/(func.L_cath + func.L_sep))*dx

plt.plot(time/3600, Li_mass)
plt.ylabel(f'Total {var[0]} Concentration [$mol/m^3$]')
plt.xlabel('Time [h]')
plt.show()

## Calculate Sulfur Species Mass Conservation
sulfur_species = overall_array[1:7, :]
sulfur_mass = np.zeros(len(sulfur_species[0]))

n_array = [8, 8, 6, 4, 2, 1] ## The number of sulfur molecules in species (following order)

for i in range(len(sulfur_species[0])):
    for j in range(len(n_array)):
        sulfur_mass[i] += n_array[j]*(np.sum(sulfur_species[j, i]))*(1/(func.L_cath + func.L_sep))*dx
        
plt.plot(time/3600, sulfur_mass)
plt.ylabel('Total Sulfur Mass Equivalent [$mol/m^3$]')
plt.xlabel('Time [h]')
#plt.savefig('{Sulfur.png', dpi=500, bbox_inches='tight')
plt.show()

## Plot all species spatial average variatian with time in the cathode (&Li2S pore volume fraction)
all_species = overall_array[0:8, :, func.index_sep_cath_boundary+1:]
Li2s = overall_array[12, :, func.index_sep_cath_boundary+1:]
fig, ax1 = plt.subplots()
for i in range(len(all_species)):
    all_temp = np.zeros(len(all_species[0]))
    Li2s_temp = np.zeros(len(all_species[0]))
    for j in range(len(all_species[0])):
        all_temp[j] = np.sum(all_species[i, j])*(1/func.L_cath)*dx
        Li2s_temp[j] = np.sum(Li2s[j])*(1/func.L_cath)*dx
        
    cutoff_frequency = 0.001  # Adjust this value based on your data
    average_time_difference = np.mean(np.diff(time))
    sampling_rate = 1 / average_time_difference
    #all_temp = butter_lowpass_filter(all_temp, cutoff_frequency, sampling_rate)
    ## Plot both on same axes
    line1 = ax1.plot(time/3600, all_temp, label=f'{var[i]}')
    ax1.set_yscale("log")
    ax1.set_ylabel('$C_{i,avg}$ in Cathode [$mol/m^3$]')
    ## Plot Li2S 
    if i == 7:
        ax2 = ax1.twinx()
        line2, = ax2.plot(time/3600, Li2s_temp, linestyle='dashed', label='$\epsilon_{Li2s, cath}$', color='y')
        color = line2.get_color()
        ax2.spines['right'].set_color(color)     # Right spine
        ax2.yaxis.label.set_color(color)         # Y2-axis label
        ax2.tick_params(axis='y', colors=color)   # Y2-axis ticks
    plt.xlabel('Time [h]')
    # Combine the handles and labels from both plots
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles += handles2
    labels += labels2
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)
plt.title('Voltage Cycle Current: 0.25A (Sine)')
#plt.savefig('Species.png', dpi=1500, bbox_inches='tight')
plt.show()

# =============================================================================
# Test flux continuity of species in separator/cathode interface
# =============================================================================

## Function for plotting
def calc_flux(s_array, phi2_array, e_cath_array, e_sep_array):

    F = 96485.3321233100184
    R = 8.3145
    T = 292.15
    x = func.x_space
    dx = x[1] - x[0]
    index_sep_cath_boundary = func.index_sep_cath_boundary

    z_Li = 1
    z_s8 = 0
    z_s8_2 = -2
    z_s6 = -2
    z_s4 = -2
    z_s2 = -2
    z_s1 = -2
    z_An = -1
    z = [z_Li, z_s8, z_s8_2, z_s6, z_s4, z_s2, z_s1, z_An]
    
    ## Equations to calculate the fluxes
    sep_flux = []
    cath_flux = []
    for i in range(len(s_array)):
        sep_flux.append([])
        cath_flux.append([])
        for j in range(len(s_array[i])):
            ## Dynamic coefficients at cathode:
            D_Li_cath = D_Li*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_cath = D_s8*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_cath = D_s8_2*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s6_cath = D_s6*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s4_cath = D_s4*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s2_cath = D_s2*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s1_cath = D_s1*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_An_cath = D_An*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_cath = [D_Li_cath, D_s8_cath, D_s8_2_cath, D_s6_cath, D_s4_cath, D_s2_cath, D_s1_cath, D_An_cath]
            ## Dynamic coefficients at seperator:
            D_Li_sep = D_Li*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_sep = D_s8*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_sep = D_s8_2*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s6_sep = D_s6*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s4_sep = D_s4*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s2_sep = D_s2*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s1_sep = D_s1*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_An_sep = D_An*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_sep = [D_Li_sep, D_s8_sep, D_s8_2_sep, D_s6_sep, D_s4_sep, D_s2_sep, D_s1_sep, D_An_sep]
            ## Variables
            curr = s_array[i][j][index_sep_cath_boundary]
            prev = s_array[i][j][index_sep_cath_boundary - 1]
            post = s_array[i][j][index_sep_cath_boundary + 1]
            phi2 = phi2_array[j][index_sep_cath_boundary]
            phi2_prevx = phi2_array[j][index_sep_cath_boundary - 1]
            phi2_postx = phi2_array[j][index_sep_cath_boundary + 1]
            ## Flux equation
            N_sep_xs = -D_sep[i]*(curr - prev)/(dx) - ((z[i]*D_sep[i]*F)/(R*T))*(curr)*(phi2 - phi2_prevx)/(dx)
            N_cath_xs = -D_cath[i]*(post - curr)/(dx) - ((z[i]*D_cath[i]*F)/(R*T))*(curr)*(phi2_postx - phi2)/(dx)
            sep_flux[i].append(N_sep_xs)
            cath_flux[i].append(N_cath_xs)
        
        ## Plot fluxes against time
        cutoff_frequency = 0.00075  # Adjust this value based on your data
        average_time_difference = np.mean(np.diff(time))
        sampling_rate = 1 / average_time_difference
        #sep_flux[i] = butter_lowpass_filter(sep_flux[i], cutoff_frequency, sampling_rate)
        #cath_flux[i] = butter_lowpass_filter(cath_flux[i], cutoff_frequency, sampling_rate)
        plt.plot(abs(I)*time/3600, sep_flux[i], label=f'{var[i]} sep flux')
        plt.plot(abs(I)*time/3600, cath_flux[i], label=f'{var[i]} cath flux', linestyle='dashdot')
        plt.plot(abs(I)*time/3600, np.array(cath_flux[i]) - np.array(sep_flux[i]), label=f'{var[i]} (cath - sep) flux', linestyle='dashed')
        plt.xlabel('Discharge Capacity [Ah]')
        plt.ylabel(f'{var[i]} FLux [$mol/(s.m^2)$]')
        plt.title(f'{var[i]} Flux vs Time at Separator/Cathode Interface')
        plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
        #plt.savefig(f'{var2[i]}_flux.png', dpi=500, bbox_inches='tight')
        plt.show()

    ## Plot electrolyte currents in separator and cathode at interface
    ie_sep = []
    ie_cath = []
    for i in range(len(sep_flux[0])):
        ie_sep.append(0)
        ie_cath.append(0)
        for j in range(len(sep_flux)):
            ie_sep[i] += F*z[j]*sep_flux[j][i]
            ie_cath[i] += F*z[j]*cath_flux[j][i]
            
    ## Plot currents
    cutoff_frequency = 0.00075  # Adjust this value based on your data
    average_time_difference = np.mean(np.diff(time))
    sampling_rate = 1 / average_time_difference
    #ie_sep = butter_lowpass_filter(ie_sep, cutoff_frequency, sampling_rate)
    #ie_cath = butter_lowpass_filter(ie_cath, cutoff_frequency, sampling_rate)
    
    I_app = I ## Applied current
    A = func.A
    I_array = (I_app/A)*np.ones(len(ie_sep))
    plt.plot(abs(I)*time/3600, ie_sep, label='$i_{e,sep}$')
    plt.plot(abs(I)*time/3600, ie_cath, label='$i_{e,cath}$', linestyle='dashed')
    plt.plot(abs(I)*time/3600, I_array, label='$I_{app}/A$')
    plt.xlabel('Discharge Capacity [Ah]')
    plt.ylabel('Current [$A/m^2$]')
    #plt.ylim([0,5])
    plt.title('$i_e$ Current vs Time at Separator/Cathode Interface')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    #plt.savefig('ie_currents.png', dpi=500, bbox_inches='tight')
    plt.show()
    
## Variables and pass into function
s_array = overall_array[0:8, :]
phi2_array = overall_array[-1, :]
e_cath_array = overall_array[8]
e_sep_array = overall_array[9]

calc_flux(s_array, phi2_array, e_cath_array, e_sep_array)