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
# Test fluxes of species in Cathode interface boundary
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
            #post = s_array[i][j][index_sep_cath_boundary + 1]
            phi2 = phi2_array[j][index_sep_cath_boundary]
            phi2_prevx = phi2_array[j][index_sep_cath_boundary - 1]
            phi1 = phi1_array[j][index_sep_cath_boundary]
            phi1_prevx = phi1_array[j][index_sep_cath_boundary - 1]
            #phi2_postx = phi2_array[j][index_sep_cath_boundary + 1]
            ## Flux equation
            N_sep_xs = -D_cath[i]*(curr - prev)/(dx) - ((z[i]*D_cath[i]*F)/(R*T))*(curr)*(phi2 - phi2_prevx)/(dx)
            #N_cath_xs = -D_cath[i]*(post - curr)/(dx) - ((z[i]*D_cath[i]*F)/(R*T))*(curr)*(phi2_postx - phi2)/(dx)
            sep_flux[i].append(N_sep_xs)
            #cath_flux[i].append(N_cath_xs)
        
        ## Plot fluxes against time
        cutoff_frequency = 0.00075  # Adjust this value based on your data
        average_time_difference = np.mean(np.diff(time))
        sampling_rate = 1 / average_time_difference
        #sep_flux[i] = butter_lowpass_filter(sep_flux[i], cutoff_frequency, sampling_rate)
        #cath_flux[i] = butter_lowpass_filter(cath_flux[i], cutoff_frequency, sampling_rate)
        plt.plot(abs(I)*time/3600, sep_flux[i], label=f'{var[i]} cath boundary flux')
        #plt.plot(abs(I)*time/3600, cath_flux[i], label=f'{var[i]} cath flux')
        #plt.plot(abs(I)*time/3600, np.array(cath_flux[i]) - np.array(sep_flux[i]), label=f'{var[i]} (cath - sep) flux')
        plt.xlabel('Discharge Capacity [Ah]')
        plt.ylabel(f'{var[i]} FLux [$mol/(s.m^2)$]')
        plt.title(f'{var[i]} Flux vs Time at Cathode Boundary')
        plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
        #plt.savefig(f'{var2[i]}_flux.png', dpi=500, bbox_inches='tight')
        plt.show()
    
    for j in range(len(s_array[i])):
        phi1 = phi1_array[j][index_sep_cath_boundary]
        phi1_prevx = phi1_array[j][index_sep_cath_boundary - 1]
        i_s.append(-1*(phi1-phi1_prevx)/dx)
    ## Plot electrolyte currents in separator and cathode at interface
    ie_sep = []
    ie_cath = []
    for i in range(len(sep_flux[0])):
        ie_sep.append(0)
        ie_cath.append(0)
        for j in range(len(sep_flux)):
            ie_sep[i] += F*z[j]*sep_flux[j][i]
            #ie_cath[i] += F*z[j]*cath_flux[j][i]
            
    ## Plot currents
    cutoff_frequency = 0.00075  # Adjust this value based on your data
    average_time_difference = np.mean(np.diff(time))
    sampling_rate = 1 / average_time_difference
    #ie_sep = butter_lowpass_filter(ie_sep, cutoff_frequency, sampling_rate)
    #ie_cath = butter_lowpass_filter(ie_cath, cutoff_frequency, sampling_rate)
    
    I_app = I ## Applied current
    A = func.A
    I_array = (I_app/A)*np.ones(len(ie_cath))
    plt.plot(abs(I)*time/3600, ie_sep, label='$i_{e}$')
    plt.plot(abs(I)*time/3600, i_s, label='$i_{s}$', linestyle='dashed')
    #plt.plot(abs(I)*time/3600, ie_cath, label='$i_{e,cath}$')
    plt.plot(abs(I)*time/3600, I_array, label='$I_{app}/A$')
    plt.xlabel('Discharge Capacity [Ah]')
    plt.ylabel('Current [$A/m^2$]')
    #plt.ylim([0,5])
    plt.title('$i_e$ Current vs Time at Cathode Boundary')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    #plt.savefig('ie&s_currents_cath.png', dpi=500, bbox_inches='tight')
    plt.show()
    
## Variables and pass into function
s_array = overall_array[0:8, :]
phi2_array = overall_array[-1, :]
phi1_array = overall_array[-2, :]
e_cath_array = overall_array[8]
e_sep_array = overall_array[9]

calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array)

# =============================================================================
# Test fluxes of species in Anode interface boundary
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
            #prev = s_array[i][j][index_sep_cath_boundary - 1]
            post = s_array[i][j][index_sep_cath_boundary + 1]
            phi2 = phi2_array[j][index_sep_cath_boundary]
            #phi2_prevx = phi2_array[j][index_sep_cath_boundary - 1]
            phi2_postx = phi2_array[j][index_sep_cath_boundary + 1]
            ## Flux equation
            #N_sep_xs = -D_sep[i]*(curr - prev)/(dx) - ((z[i]*D_sep[i]*F)/(R*T))*(curr)*(phi2 - phi2_prevx)/(dx)
            N_cath_xs = -D_sep[i]*(post - curr)/(dx) - ((z[i]*D_sep[i]*F)/(R*T))*(curr)*(phi2_postx - phi2)/(dx)
            #sep_flux[i].append(N_sep_xs)
            cath_flux[i].append(N_cath_xs)
        
        ## Plot fluxes against time
        cutoff_frequency = 0.00075  # Adjust this value based on your data
        average_time_difference = np.mean(np.diff(time))
        sampling_rate = 1 / average_time_difference
        #sep_flux[i] = butter_lowpass_filter(sep_flux[i], cutoff_frequency, sampling_rate)
        #cath_flux[i] = butter_lowpass_filter(cath_flux[i], cutoff_frequency, sampling_rate)
        #plt.plot(abs(I)*time/3600, sep_flux[i], label=f'{var[i]} cath boundary flux')
        plt.plot(abs(I)*time/3600, cath_flux[i], label=f'{var[i]} anode flux')
        #plt.plot(abs(I)*time/3600, np.array(cath_flux[i]) - np.array(sep_flux[i]), label=f'{var[i]} (cath - sep) flux')
        plt.xlabel('Discharge Capacity [Ah]')
        plt.ylabel(f'{var[i]} FLux [$mol/(s.m^2)$]')
        plt.title(f'{var[i]} Flux vs Time at Anode Boundary')
        plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
        #plt.savefig(f'{var2[i]}_flux.png', dpi=500, bbox_inches='tight')
        plt.show()
    
    for j in range(len(s_array[i])):
        phi1 = phi1_array[j][index_sep_cath_boundary]
        phi1_prevx = phi1_array[j][index_sep_cath_boundary - 1]
        phi1_postx = phi1_array[j][index_sep_cath_boundary + 1]
        i_s.append(-1*(phi1_postx-phi1)/dx)
    ## Plot electrolyte currents in separator and cathode at interface
    ie_sep = []
    ie_cath = []
    for i in range(len(cath_flux[0])):
        ie_sep.append(0)
        ie_cath.append(0)
        for j in range(len(sep_flux)):
            #ie_sep[i] += F*z[j]*sep_flux[j][i]
            if j == 0:
                ie_cath[i] += F*z[j]*cath_flux[j][i]
            
    ## Plot currents
    cutoff_frequency = 0.00075  # Adjust this value based on your data
    average_time_difference = np.mean(np.diff(time))
    sampling_rate = 1 / average_time_difference
    #ie_sep = butter_lowpass_filter(ie_sep, cutoff_frequency, sampling_rate)
    #ie_cath = butter_lowpass_filter(ie_cath, cutoff_frequency, sampling_rate)
    
    I_app = I ## Applied current
    A = func.A
    I_array = (I_app/A)*np.ones(len(ie_cath))
    #plt.plot(abs(I)*time/3600, ie_sep, label='$i_{e}$')
    plt.plot(abs(I)*time/3600, i_s, label='$i_{s}$', linestyle='dashed')
    plt.plot(abs(I)*time/3600, ie_cath, label='$i_{e}$')
    plt.plot(abs(I)*time/3600, I_array, label='$I_{app}/A$')
    plt.xlabel('Discharge Capacity [Ah]')
    plt.ylabel('Current [$A/m^2$]')
    #plt.ylim([0,5])
    plt.title('$i_e$ Current vs Time at Anode Interface')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    #plt.savefig('ie&s_currents_anod.png', dpi=500, bbox_inches='tight')
    #plt.show()
    
## Variables and pass into function
s_array = overall_array[0:8, :]
phi2_array = overall_array[-1, :]
phi1_array = overall_array[-2, :]
e_cath_array = overall_array[8]
e_sep_array = overall_array[9]

calc_flux(s_array, phi2_array, phi1_array, e_cath_array, e_sep_array)

# =============================================================================
# Plot i1 at anode (Checking)
# =============================================================================
Li = overall_array[0, :]
phi2 = overall_array[-1, :]
e = overall_array[9]
T = 292.15
F = 96485.3321233100184
R = 8.3145
i1 = []
for i in range(0, len(Li)):
    post = Li[i][1]
    curr = Li[i][0]
    phi2_postx = phi2[i][1]
    phi2_curr = phi2[i][0]
    e_curr = e[i][0]**1.5
    eta1 = 0 - phi2_curr - 0
    i1_Li = 0.3*(np.exp(0.5*(F/(R*T))*eta1) - (curr/1001)*np.exp(-0.5*(F/(R*T))*eta1)) 
    i1.append(i1_Li)

i1 = np.array(i1)
plt.plot(abs(I)*time/3600, i1, label='$i_{1}$', linestyle='dashed')
plt.xlabel('Discharge Capacity [Ah]')
plt.ylabel('Current [$A/m^2$]')
#plt.ylim([0,5])
plt.title('$i_e$ Current vs Time at Anode Interface')
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
#plt.savefig('ie&s_currents_anod.png', dpi=500, bbox_inches='tight')
plt.show()