import numpy as np 
import matplotlib.pyplot as plt
import generate_square_wave

## We will evaluate the ohmic resistance here
data = np.load('var_array_0.25A_sine.npz', allow_pickle=True)
overall_array = data['solved'][:, :-1]
time = data['time'][:-1]

## Evaluate current profile
start_time = 0
end_time = 25000
high_value = 0.25 ## Current value
low_value = 0
high_pulse_length = (end_time/75)*0.7
low_pulse_length = (end_time/75)*0.3
freq = 2*np.pi*0.00025 ## 2.5e-4 Hz
high_time = 21000
low_time = 34000
charge_curr = -0.4
curr = []
for i in range(len(time)):
    #curr.append(generate_square_wave.ret_val_square(time[i], start_time, end_time, high_value, low_value, high_pulse_length, low_pulse_length))
    curr.append(generate_square_wave.ret_val_sin(time[i], high_value, freq))
    #curr.append(generate_square_wave.ret_val_steps(time[i], high_value, charge_curr, high_time, low_time))
    
curr = np.asarray(curr)
plt.plot(time/3600, curr, color='darkorange')
plt.title('Current Profile')
plt.xlabel('Time (h)')
plt.ylabel('Current (A)')
#plt.grid(True)
#plt.savefig('Current_Sine.png', dpi=1500, bbox_inches='tight')
plt.show()

## Evaluate voltage
phi1 = overall_array[14]
phi2 = overall_array[15]

phi1_cath = phi1[:, -1]
phi2_anod = phi2[:, 0]

voltage = phi1_cath - phi2_anod 
plt.plot(time/3600, voltage)
plt.title(f'Voltage Cycle Current: {round(high_value, 3)}A')
plt.ylabel('Cell Voltage [V]')
plt.xlabel('Time [h]')
plt.ylim([2.35,2.5])
#plt.savefig('Voltage_Sine.png', dpi=1500, bbox_inches='tight')
plt.show()

# =============================================================================
# Combine_Current and Voltage plots
# =============================================================================
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
line1, = ax1.plot(time/3600, voltage)
ax1.set_ylabel('Cell Voltage [V]')
color1 = line1.get_color()
ax1.spines['left'].set_color(color1)     # Right spine
ax1.yaxis.label.set_color(color1)         # Y2-axis label
ax1.tick_params(axis='y', colors=color1)   # Y2-axis ticks
ax1.set_ylim([2.35,2.5])
ax1.set_xlabel('Time [h]')

line2,  = ax2.plot(time/3600, curr, color='darkorange', linestyle='dashed')
ax2.set_ylabel('Current [A]')
color2 = line2.get_color()
ax2.spines['right'].set_color(color2)     # Right spine
ax2.yaxis.label.set_color(color2)         # Y2-axis label
ax2.tick_params(axis='y', colors=color2)   # Y2-axis ticks
plt.title('Voltage Cycle Current: 0.25A (Sine)')
ax2.set_xlabel('Time [h]')
#plt.savefig('Voltage_Sine.png', dpi=1500, bbox_inches='tight')
plt.show()

## Evaluate ohmic resistance
ohm = []
for i in range(1, len(voltage)):
    epsilon = 1e-8
    dI = curr[i] - curr[i-1]
    dV = voltage[i] - voltage[i-1]
    R = dV/(dI + epsilon)
    ohm.append(R)

ohm = np.asarray(ohm)
# =============================================================================
# plt.plot(time[1:]/3600, ohm)
# plt.ylabel('Ohmic Resistance [$\Omega$]')
# plt.xlabel('Time [h]')
# =============================================================================

## Try performing impedence calculations (EIS)

# Perform Fourier Transform to convert data to frequency domain
frequency_spectrum_current = np.fft.fft(curr)
frequency_spectrum_voltage = np.fft.fft(voltage)

# Calculate impedance in frequency domain
impedance = frequency_spectrum_voltage / frequency_spectrum_current

# Nyquist Plot
plt.figure(figsize=(8, 6))
plt.plot(np.real(impedance), -np.imag(impedance), marker='o')
plt.xlabel('Real(Z)')
plt.ylabel('-Imaginary(Z)')
plt.title('Nyquist Plot')
plt.grid(True)
plt.show()

# Bode Plot
magnitude = np.abs(impedance)
phase = np.angle(impedance, deg=True)  # Convert phase to degrees

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogx(np.fft.fftfreq(len(impedance)), magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Z|')
plt.title('Bode Plot - Magnitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(np.fft.fftfreq(len(impedance)), phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.title('Bode Plot - Phase')
plt.grid(True)

plt.tight_layout()
plt.show()