import numpy as np 
import matplotlib.pyplot as plt
import func_1D_LiS as func
from scipy.signal import butter, filtfilt

## Low Pass filter function to filter noise from data    
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

data = np.load('var_array_0.5A_constant.npz', allow_pickle=True)
res = abs(data['residuals'])
row_labels = func.row_label
time = data['time']/3600

row_label = []
for i in range(len(row_labels)):
    row_label.append([i, str(row_labels[i][0])])
row_label = np.asarray(row_label)

## We will plot the residual function avg plot for each speceies over time
var = ['$Li^+$', '$s_{8(l)}$', '${s_8}^{2-}$', '${s_6}^{2-}$', '${s_4}^{2-}$', '${s_2}^{2-}$', '$s^{2-}$', '$A^{-}$']
poros_list = ['$\epsilon_{cath}$', '$\epsilon_{sep}$', '$\epsilon_{s8s, cath}$', '$\epsilon_{s8s, sep}$', '$\epsilon_{Li2s, cath}$', '$\epsilon_{Li2s, sep}$', '$\phi_1$', '$\phi_2$']
variables = var + poros_list

avg_res = [] ## To store average residual values at each time step
vars_label = np.unique(row_label[:, 1])
for i in range(len(res)):
    temp = []
    for j in range(len(vars_label)):
        avg_var_res = np.sum(res[i][row_label[:, 1]==vars_label[j]])/len(res[i][row_label[:, 1]==vars_label[j]])
        temp.append(avg_var_res)
    avg_res.append(temp)
    
## Plot the residuals
avg_res = np.asarray(avg_res)
index_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] ## variables to plot index
for i in index_to_plot:
    if i%2 == 0:
        linestyle = 'dashed'
    else:
        linestyle = 'solid'
    plt.plot(time[1:], avg_res[:, i], label=f'{variables[i]}', linestyle=linestyle)

plt.yscale('log')
plt.xlabel('Time [h]')
plt.ylabel('Variable Residual')
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
#plt.savefig('residuals.png', dpi=300, bbox_inches='tight')