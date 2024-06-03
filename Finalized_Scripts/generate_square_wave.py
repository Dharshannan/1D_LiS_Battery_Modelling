import numpy as np
import matplotlib.pyplot as plt

## Function to return value at certain time step
## Function for a square wave
def ret_val_square(time, start_time, end_time, high_value, low_value, high_pulse_length, low_pulse_length): 
    duration = end_time - start_time
    period = high_pulse_length + low_pulse_length
    phase = (time - start_time) % period
    
    if phase < high_pulse_length:
        return high_value
    else:
        return low_value

## Function for a sinusoidal wave
def ret_val_sin(time, I_app, freq):
    return(I_app*np.sin(time*freq))

## Function for discharge-rest-charge
def ret_val_steps(time, high_value, low_value, high_time, low_time):
    ## Discharge
    if time <= high_time:
        return(high_value)
    ## Charge
    if time >= low_time:
        return(low_value)
    ## Rest
    else:
        return(0)

# =============================================================================
# ### Below are for testing ###
# # Parameters
# start_time = 0
# end_time = 25000 #7*3600
# high_value = 0.5
# low_value = -0.4
# high_pulse_length = (end_time/50)*0.7
# low_pulse_length = (end_time/50)*0.3
# freq = 2*np.pi*0.00025
# high_time = 21600
# low_time = 34200
# 
# # Value at specific time
# time = 2.95
# #value_at_time = ret_val(time, start_time, end_time, high_value, low_value, high_pulse_length, low_pulse_length)
# #value_at_time = ret_val_sin(time, high_value, freq)
# value_at_time = ret_val_steps(time, high_value, low_value, high_time, low_time)
# print(f"Value of the square wave at time {time}: {value_at_time}")
# 
# time_interval = np.linspace(0, 75000, 250)
# curr = []
# for i in range(len(time_interval)):
#     #curr.append(ret_val(time_interval[i], start_time, end_time, high_value, low_value, high_pulse_length, low_pulse_length))
#     curr.append(ret_val_steps(time_interval[i], high_value, low_value, high_time, low_time))
#     
# curr = np.asarray(curr)
# 
# plt.plot(time_interval, curr)
# plt.title('GITT/EIS Current')
# plt.xlabel('Time (s)')
# plt.ylabel('Current (A)')
# #plt.xlim([0, 10])
# #plt.grid(True)
# #plt.savefig('Current_GITT.png', dpi=1500, bbox_inches='tight')
# plt.show()
# =============================================================================
