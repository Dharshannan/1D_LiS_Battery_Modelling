import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter, medfilt

## Low Pass filter function to filter noise from data    
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

## We will test the Gustafsson adaptive time step approach on the residuals that we have
## already obtained
data = np.load('var_array_0.5A_constant.npz', allow_pickle=True)
res = abs(data['residuals'])
overall_array = data['solved']
time = data['time']

## Calculate the max(R) array and h array
R = []
for i in range(len(res)):
    R.append(max(res[i]))
    
R = np.asarray(R)
## Remove noise from residual array due to noise from numerical instabilities
cutoff_frequency = 0.05  # Adjust this value based on your data
average_time_difference = np.mean(np.diff(time))
sampling_rate = 1 / average_time_difference
# Apply Savitzky-Golay filter
window_length = 50  # window size for filtering
poly_order = 10   # polynomial order
# Apply median filter
#R = medfilt(R, kernel_size=155) ## >151 (*Odd number) seems to work well here
"""
kernel_size here is the number of neighboring points (including each data point)
considered when calculating the median at each point for filtering/denoising.
"""
#R = savgol_filter(R, window_length, poly_order)
#R = butter_lowpass_filter(R, cutoff_frequency, sampling_rate)
# =============================================================================
# NOTE: Median filter seems to work well here
# =============================================================================
#plt.plot(time[1:]/3600, R)
plt.plot(R)
plt.xlabel('Iterations')
plt.ylabel('Max Residuals')
plt.yscale('log')
plt.show()

# Voltage plot
species1 = overall_array[:, :]
phi1 = species1[14]
phi2 = species1[15]

phi1_cath = phi1[:, -1]
phi2_anod = phi2[:, 0]
voltage = phi1_cath - phi2_anod 
#voltage = medfilt(voltage, kernel_size=151)

# =============================================================================
# Calculate and plot gradient ratio of residuals
# =============================================================================
res_grads = []
res_grads_ratio = []
max_res = 0
for i in range(1, len(R)):
    R_temp = R[:i+1]
    #print(i)
    if i >=100:
        # Apply median filter to remove noise
        #R_temp = medfilt(R_temp, kernel_size=3) ## >151 (*Odd number) seems to work well here
        R_temp = list(R_temp) ## Convert back to Python list

        temp1 = (R_temp[i] - R_temp[i-1])
        res_grads.append(temp1)
        max_res = max(temp1, max_res)
        #temp2 = temp1/(max(res_grads) + 1e-10)
        temp2 = temp1/(max_res + 1e-10)
        res_grads_ratio.append(temp2)
        
    else:
        res_grads.append(0)
        res_grads_ratio.append(0)

plt.plot(res_grads_ratio)
res_grads_ratio = np.array(res_grads_ratio)
val = 1e-5
x_vals = np.array(range(len(res_grads_ratio)))
plt.scatter(x_vals[res_grads_ratio>val], res_grads_ratio[res_grads_ratio>val], c='r')
plt.xlabel('Iterations')
plt.ylabel('Residual Gradient Ratio')
#plt.yscale('log')
plt.show()
## Lets try to make aanother step size condition using this ratios ##
plt.plot(voltage)
volt = voltage[2:]
volt_x = np.array(range(len(voltage[2:])))
plt.scatter(volt_x[res_grads_ratio>val], volt[res_grads_ratio>val], c='r', label='Step size reduction required')
plt.xlabel('Iterations')
plt.ylabel('Voltage [V]')
plt.legend()
plt.show()