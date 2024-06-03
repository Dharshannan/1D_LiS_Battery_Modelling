import numpy as np 
import matplotlib.pyplot as plt
import func_potentials as func
from scipy.signal import butter, filtfilt

# Define a custom list of colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                 '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

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
# Test fluxes of A- in Cathode interface boundary
# =============================================================================

## Function for plotting
def calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array):

    F = 96485.3321233100184
    R = 8.3145
    T = 292.15
    x = func.x_space
    dx = x[1] - x[0]
    index_sep_cath_boundary = -1

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
    i_s = []
    for i in range(len(s_array)):
        sep_flux.append([])
        cath_flux.append([])
        #i_s.append([])
        for j in range(len(s_array[i])):
            ## Dynamic coefficients at cathode:
            D_Li_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_cath = 0.88e-11*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s6_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s4_cath = 1.75e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s2_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s1_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_An_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_cath = [D_Li_cath, D_s8_cath, D_s8_2_cath, D_s6_cath, D_s4_cath, D_s2_cath, D_s1_cath, D_An_cath]
            ## Dynamic coefficients at seperator:
            D_Li_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_sep = 0.88e-11*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s6_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s4_sep = 1.75e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s2_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s1_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_An_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_sep = [D_Li_sep, D_s8_sep, D_s8_2_sep, D_s6_sep, D_s4_sep, D_s2_sep, D_s1_sep, D_An_sep]
            ## Variables
            curr = s_array[i][j][index_sep_cath_boundary]
            prev = s_array[i][j][index_sep_cath_boundary - 1]
            #post = s_array[i][j][index_sep_cath_boundary + 1]
            phi2 = phi2_array[j][index_sep_cath_boundary]
            phi2_prevx = phi2_array[j][index_sep_cath_boundary - 1]
            phi1 = phi1_array[j][index_sep_cath_boundary]
            phi1_prevx = phi1_array[j][index_sep_cath_boundary - 1]
            #phi2_postx = phi2_array[j][index_sep_cath_boundary + 1]
            ## Flux equation
            N_sep_xs = -D_sep[i]*(curr - prev)/(dx) - ((z[i]*D_sep[i]*F)/(R*T))*(curr)*(phi2 - phi2_prevx)/(dx)
            #N_cath_xs = -D_cath[i]*(post - curr)/(dx) - ((z[i]*D_cath[i]*F)/(R*T))*(curr)*(phi2_postx - phi2)/(dx)
            sep_flux[i].append(N_sep_xs)
            #cath_flux[i].append(N_cath_xs)
        
        ## Plot fluxes against time
        cutoff_frequency = 0.00075  # Adjust this value based on your data
        average_time_difference = np.mean(np.diff(time))
        sampling_rate = 1 / average_time_difference
        #sep_flux[i] = butter_lowpass_filter(sep_flux[i], cutoff_frequency, sampling_rate)
        #cath_flux[i] = butter_lowpass_filter(cath_flux[i], cutoff_frequency, sampling_rate)
        if i == 7:
            plt.plot(abs(I)*time/3600, sep_flux[i], label=f'{var[i]} cath boundary flux', c='black')
        #plt.plot(abs(I)*time/3600, cath_flux[i], label=f'{var[i]} cath flux')
        #plt.plot(abs(I)*time/3600, np.array(cath_flux[i]) - np.array(sep_flux[i]), label=f'{var[i]} (cath - sep) flux')

    
## Variables and pass into function
s_array = overall_array[0:8, :]
phi2_array = overall_array[-1, :]
phi1_array = overall_array[-2, :]
e_cath_array = overall_array[8]
e_sep_array = overall_array[9]

#calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array)

# =============================================================================
# Test fluxes of A- in Anode interface boundary
# =============================================================================

## Function for plotting
def calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array):

    F = 96485.3321233100184
    R = 8.3145
    T = 292.15
    x = func.x_space
    dx = x[1] - x[0]
    index_sep_cath_boundary = 0
    
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
    i_s = []
    for i in range(len(s_array)):
        sep_flux.append([])
        cath_flux.append([])
        #i_s.append([])
        for j in range(len(s_array[i])):
            ## Dynamic coefficients at cathode:
            D_Li_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_cath = 0.88e-11*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s6_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s4_cath = 1.75e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s2_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s1_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_An_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_cath = [D_Li_cath, D_s8_cath, D_s8_2_cath, D_s6_cath, D_s4_cath, D_s2_cath, D_s1_cath, D_An_cath]
            ## Dynamic coefficients at seperator:
            D_Li_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_sep = 0.88e-11*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s6_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s4_sep = 1.75e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s2_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s1_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_An_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_sep = [D_Li_sep, D_s8_sep, D_s8_2_sep, D_s6_sep, D_s4_sep, D_s2_sep, D_s1_sep, D_An_sep]
            ## Variables
            curr = s_array[i][j][index_sep_cath_boundary]
            #prev = s_array[i][j][index_sep_cath_boundary - 1]
            post = s_array[i][j][index_sep_cath_boundary + 1]
            phi2 = phi2_array[j][index_sep_cath_boundary]
            #phi2_prevx = phi2_array[j][index_sep_cath_boundary - 1]
            phi2_postx = phi2_array[j][index_sep_cath_boundary + 1]
            ## Flux equation
            #N_sep_xs = -D_sep[i]*(curr - prev)/(dx) - ((z[i]*D_sep[i]*F)/(R*T))*(curr)*(phi2 - phi2_prevx)/(dx)
            N_cath_xs = -D_cath[i]*(post - curr)/(dx) - ((z[i]*D_cath[i]*F)/(R*T))*(curr)*(phi2_postx - phi2)/(dx)
            #sep_flux[i].append(N_sep_xs)
            cath_flux[i].append(N_cath_xs)
        
        ## Plot fluxes against time
        cutoff_frequency = 0.00075  # Adjust this value based on your data
        average_time_difference = np.mean(np.diff(time))
        sampling_rate = 1 / average_time_difference
        #sep_flux[i] = butter_lowpass_filter(sep_flux[i], cutoff_frequency, sampling_rate)
        #cath_flux[i] = butter_lowpass_filter(cath_flux[i], cutoff_frequency, sampling_rate)
        #plt.plot(abs(I)*time/3600, sep_flux[i], label=f'{var[i]} cath boundary flux')
        if i == 7:
            plt.plot(abs(I)*time/3600, cath_flux[i], label=f'{var[i]} anode boundary flux')
        #plt.plot(abs(I)*time/3600, np.array(cath_flux[i]) - np.array(sep_flux[i]), label=f'{var[i]} (cath - sep) flux')

    
## Variables and pass into function
s_array = overall_array[0:8, :]
phi2_array = overall_array[-1, :]
phi1_array = overall_array[-2, :]
e_cath_array = overall_array[8]
e_sep_array = overall_array[9]

#calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array)

# =============================================================================
# Test fluxes of A- in other points
# =============================================================================

## Function for plotting
def calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array, index_to_plot, linestyle, colors):

    F = 96485.3321233100184
    R = 8.3145
    T = 292.15
    x = func.x_space
    dx = x[1] - x[0]
    index_sep_cath_boundary = index_to_plot
    
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
    i_s = []
    for i in range(len(s_array)):
        sep_flux.append([])
        cath_flux.append([])
        #i_s.append([])
        for j in range(len(s_array[i])):
            ## Dynamic coefficients at cathode:
            D_Li_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_cath = 0.88e-11*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s6_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s4_cath = 1.75e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s2_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_s1_cath = 0.88e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_An_cath = 3.5e-12*(e_cath_array[j][index_sep_cath_boundary]**1.5)
            D_cath = [D_Li_cath, D_s8_cath, D_s8_2_cath, D_s6_cath, D_s4_cath, D_s2_cath, D_s1_cath, D_An_cath]
            ## Dynamic coefficients at seperator:
            D_Li_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_sep = 0.88e-11*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s8_2_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s6_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s4_sep = 1.75e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s2_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_s1_sep = 0.88e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_An_sep = 3.5e-12*(e_sep_array[j][index_sep_cath_boundary]**1.5)
            D_sep = [D_Li_sep, D_s8_sep, D_s8_2_sep, D_s6_sep, D_s4_sep, D_s2_sep, D_s1_sep, D_An_sep]
            ## Variables
            curr = s_array[i][j][index_sep_cath_boundary]
            #prev = s_array[i][j][index_sep_cath_boundary - 1]
            post = s_array[i][j][index_sep_cath_boundary + 1]
            phi2 = phi2_array[j][index_sep_cath_boundary]
            #phi2_prevx = phi2_array[j][index_sep_cath_boundary - 1]
            phi2_postx = phi2_array[j][index_sep_cath_boundary + 1]
            ## Flux equation
            #N_sep_xs = -D_sep[i]*(curr - prev)/(dx) - ((z[i]*D_sep[i]*F)/(R*T))*(curr)*(phi2 - phi2_prevx)/(dx)
            N_cath_xs = -D_cath[i]*(post - curr)/(dx) - ((z[i]*D_cath[i]*F)/(R*T))*(curr)*(phi2_postx - phi2)/(dx)
            #sep_flux[i].append(N_sep_xs)
            cath_flux[i].append(N_cath_xs)
        
        ## Plot fluxes against time
        cutoff_frequency = 0.00075  # Adjust this value based on your data
        average_time_difference = np.mean(np.diff(time))
        sampling_rate = 1 / average_time_difference
        #sep_flux[i] = butter_lowpass_filter(sep_flux[i], cutoff_frequency, sampling_rate)
        #cath_flux[i] = butter_lowpass_filter(cath_flux[i], cutoff_frequency, sampling_rate)
        #plt.plot(abs(I)*time/3600, sep_flux[i], label=f'{var[i]} cath boundary flux')
        if i == 7:
            plt.plot(abs(I)*time/3600, cath_flux[i], label=f'{var[i]} point {index_to_plot} flux', linestyle=linestyle, c=colors)
        #plt.plot(abs(I)*time/3600, np.array(cath_flux[i]) - np.array(sep_flux[i]), label=f'{var[i]} (cath - sep) flux')

    
## Variables and pass into function
s_array = overall_array[0:8, :]
phi2_array = overall_array[-1, :]
phi1_array = overall_array[-2, :]
e_cath_array = overall_array[8]
e_sep_array = overall_array[9]

for i in range(14, 24):
    if i <= func.index_sep_cath_boundary:
        calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array, i, 'dashed', colors[i])
        
    else:
        calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array, i, 'solid', colors[i-14])
    
plt.xlabel('Discharge Capacity [Ah]')
plt.ylabel(f'{var[7]} FLux [$mol/(s.m^2)$]')
#plt.yscale('log')
plt.title(f'{var[7]} Flux vs Time at all points')
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
#plt.savefig(f'{var2[7]}_flux_2.png', dpi=500, bbox_inches='tight')
plt.show()
