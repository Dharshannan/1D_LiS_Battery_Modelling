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

h = []
for i in range(1, len(time)):
    h.append(time[i] - time[i-1])
    
h = np.asarray(h)
h_old = h.copy()

## Now we apply the Gustaffson approach
## k1 & k2 values that tested to be good (add to below):
## (k1=2, k2=0.001), 
def upd_step(R, h, i):
    k = 1
    k1 = 2
    k2 = 0.001
    tol = 1e-2
    denom = R[i-1] + 1e-10
    
    if i == 1:
        h_new = ((tol/denom)**(1/k))*(h[i-1])
        
    else:
        h_new = (h[i-1]/h[i-2])*((tol/denom)**(k2/k))*((R[i-2]/denom)**(k1/k))*h[i-1]
    
    return(h_new)

for i in range(1, len(h)):
    max_h = 5
    min_h = 1e-1
    h[i] = upd_step(R, h, i)
    if h[i] < min_h:
        h[i] = min_h
    if h[i] > max_h:
        h[i] = max_h

## Plot step sizes
fig, ax1 = plt.subplots()
#ax1.plot(time[1:]/3600, h, c='steelblue')
ax1.plot(h, c='steelblue')
ax1.set_yscale('log')
ax1.set_ylabel('Step Size [s]', c='steelblue')
ax1.set_xlabel('Iterations')

# Voltage plot
species1 = overall_array[:, :]
phi1 = species1[14]
phi2 = species1[15]

phi1_cath = phi1[:, -1]
phi2_anod = phi2[:, 0]
voltage = phi1_cath - phi2_anod 
#voltage = medfilt(voltage, kernel_size=151)

ax2 = ax1.twinx()
ax2.plot(voltage, c='darkorange', linestyle='dashed')
#ax2.plot(time[:]/3600, voltage, c='darkorange', linestyle='dashed')
ax2.set_ylabel('Cell Voltage [V]', c='darkorange')
ax2.set_xlabel('Iterations')
ax2.set_ylim([1.9,2.5])

ax1.spines['left'].set_color('steelblue')     # Left spine
ax2.spines['right'].set_color('darkorange')     # Right spine
ax1.yaxis.label.set_color('steelblue')        # Y1-axis label
ax2.yaxis.label.set_color('darkorange')         # Y2-axis label
ax1.tick_params(axis='y', colors='steelblue')  # Y1-axis ticks
ax2.tick_params(axis='y', colors='darkorange')   # Y2-axis ticks
plt.show()

## Comparison plot of step size
plt.plot(h_old, label='Old step size')
plt.plot(h, label='New step size')
plt.xlabel('Iterations')
plt.ylabel('Step Size [s]')
plt.yscale('log')
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
plt.show()