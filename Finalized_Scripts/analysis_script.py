import numpy as np 
import matplotlib.pyplot as plt
import func_potentials as func
from scipy.signal import butter, filtfilt

## Load Saved Data ##
data = np.load('var_array_0.5A_constant.npz', allow_pickle=True)
charge_start = -1 ## Index to start charge
overall_array = data['solved'][:, :charge_start]
time = data['time'][:charge_start]
runtime = data['runtime']/3600
print('Runtime: ', runtime, 'h')
print('Charge start residual (max): ', max(data['residuals'][charge_start]))
#jacob_array = data['jacobs']
#print(max(jacob_array))
index_sep_cath_boundary = func.index_sep_cath_boundary

x = func.x_space
var = ['$Li^+$', '$s_{8(l)}$', '${s_8}^{2-}$', '${s_6}^{2-}$', '${s_4}^{2-}$', '${s_2}^{2-}$', '$s^{2-}$', '$A^{-}$', '$\phi_1$', '$\phi_2$']
var2 = ['Li+', 's8l', 's8_2-', 's6_2-', 's4_2-', 's2_2-', 's_2-', 'A-', 'phi1', 'phi2']

species1 = overall_array[:, :]
print(species1.shape)
species = np.concatenate((species1[:8, :, :], species1[14:, :, :]))
time = time[:]
I = 0.5 ## Current value used
Ah = time*I/3600
#print(len(species[0]))

for i in range(len(var)):
    for j in range(0, len(species[i]), int(species1.shape[1]/10)):
        plt.plot(x, species[i][j], label=f'time: {round(time[j], 0)}')
        plt.scatter(x[index_sep_cath_boundary], species[i][j][index_sep_cath_boundary], marker='o')
    
    if i == len(var) - 1 or i == len(var) - 2:
        plt.ylabel(f'{var[i]} Potential [V]')

    else:
        plt.yscale("log")
        plt.ylabel(f'{var[i]} Concentration [$mol/m^3$]')
    plt.xlabel('Cell Distance [m]')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    #plt.savefig(f'{var2[i]}_concentration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# Voltage plot
phi1 = species1[14]
phi2 = species1[15]

phi1_cath = phi1[:, -1]
phi2_anod = phi2[:, 0]

voltage = phi1_cath - phi2_anod   
## Low Pass filter function to filter noise from data    
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
cutoff_frequency = 0.001  # Adjust this value based on your data
average_time_difference = np.mean(np.diff(time))
sampling_rate = 1 / average_time_difference
#voltage = butter_lowpass_filter(voltage, cutoff_frequency, sampling_rate)

plt.plot(Ah, voltage)
plt.title(f'Voltage Discharge Current: {round(I, 3)}A')
plt.ylabel('Cell Voltage [V]')
plt.xlabel('Discharge Capacity [Ah]')
plt.ylim([1.925,2.5])
#plt.savefig('Voltage_GITT.png', dpi=1500, bbox_inches='tight')
plt.show()

## Plot porosity
poros_list = ['$\epsilon_{cath}$', '$\epsilon_{sep}$', '$\epsilon_{s8s, cath}$', '$\epsilon_{s8s, sep}$', '$\epsilon_{Li2s, cath}$', '$\epsilon_{Li2s, sep}$']
poros = species1[8:14]
for i in range(6):
    for j in range(0, len(poros[i]), int(species1.shape[1]/10)):
        plt.plot(x, poros[i][j], label=f'time: {round(time[j], 0)}', marker='none')
        plt.scatter(x[index_sep_cath_boundary], poros[i][j][index_sep_cath_boundary], marker='o')
    #plt.yscale("log")
    plt.ylabel(f'Volume Fraction {poros_list[i]}')
    plt.xlabel('Cell Distance [m]')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    #plt.savefig(f'poros{i}_concentration.png', dpi=300, bbox_inches='tight')
    plt.show()
