import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 3e8  # Speed of light
f0 = 77e9  # Radar carrier frequency
B = 4e9  # Bandwidth (Hz)
T_chirp = 50e-6  # Chirp duration (s)
S = B / T_chirp  # Chirp slope (Hz/s)
num_chirps = 128  # Chirps per frame
num_samples = 256  # ADC samples per chirp
fs = num_samples / T_chirp  # Sampling rate

# Target movement
frame_rate = 10  # Hz
dt = 1 / frame_rate
target_speed = 1.0  # m/s (horizontal)
target_y = 3.0  # m
x0 = -1.0  # Initial X

# Create radar data cube (num_chirps x num_samples)
frame = 10  # mid-frame
t_chirp = np.arange(num_chirps) * T_chirp
t_sample = np.arange(num_samples) / fs
x_pos = x0 + target_speed * frame * dt
r = np.sqrt(x_pos**2 + target_y**2)
angle_rad = np.arctan2(x_pos, target_y)
vr = target_speed * np.cos(angle_rad)
fd = 2 * vr * f0 / c  # Doppler shift
tau = 2 * r / c  # Time delay

# Generate signal
data = np.zeros((num_chirps, num_samples), dtype=complex)
for i in range(num_chirps):
    for j in range(num_samples):
        t = t_sample[j]
        beat_freq = S * (t - tau)
        phase = 2 * np.pi * (beat_freq * t + fd * i * T_chirp)
        data[i, j] = np.exp(1j * phase)

# Apply 2D FFT (Range-Doppler processing)
spectrum = np.fft.fft2(data, s=(num_chirps, num_samples))
spectrum = np.fft.fftshift(spectrum, axes=0)
spectrum_mag = 20 * np.log10(np.abs(spectrum) + 1e-6)

# Plot Range-Doppler Spectrum
plt.figure(figsize=(10, 6))
extent = [0, num_samples * c / (2 * B), -frame_rate/2, frame_rate/2]
plt.imshow(spectrum_mag, aspect='auto', extent=extent, cmap='inferno', origin='lower')
plt.xlabel('Range (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Simulated FMCW Radar Range-Doppler Spectrum')
plt.colorbar(label='Magnitude (dB)')
plt.grid(True)
plt.show()


# Simulate a simple 2D MIMO array with virtual antennas (angle dimension)
num_rx = 8  # number of virtual RX antennas (for angle FFT)
angle_bins = 64  # angle FFT size

# Create a 3D data cube: (angle, doppler, range)
# We simulate by replicating the RD map across antennas with angle phase shifts
data_cube = np.zeros((num_rx, num_chirps, num_samples), dtype=complex)

# Assume target at a known angle (angle_rad)
wavelength = c / f0
d = wavelength / 2  # element spacing
for ant in range(num_rx):
    phase_shift = 2 * np.pi * d * ant * np.sin(angle_rad) / wavelength
    data_cube[ant, :, :] = data * np.exp(1j * phase_shift)

# Apply 3D FFT: Angle (across antennas), Doppler, Range
fft_cube = np.fft.fftshift(np.fft.fftn(data_cube, s=(angle_bins, num_chirps, num_samples), axes=(0,1,2)), axes=0)
fft_mag = np.abs(fft_cube)

# Find the peak response in the 3D FFT cube
peak_idx = np.unravel_index(np.argmax(fft_mag), fft_mag.shape)
angle_bin, doppler_bin, range_bin = peak_idx

# Convert bins to physical quantities
range_res = c / (2 * B)
velocity_res = c / (2 * f0 * T_chirp * num_chirps)
angle_res = 180 / angle_bins  # in degrees

det_range = range_bin * range_res
det_velocity = (doppler_bin - num_chirps // 2) * velocity_res
det_angle = (angle_bin - angle_bins // 2) * angle_res

# Display extracted values
# {
#     "Estimated Range (m)": round(det_range, 2),
#     "Estimated Velocity (m/s)": round(det_velocity, 2),
#     "Estimated Angle (deg)": round(det_angle, 2)
# }
print(f"Estimated Range (m): {round(det_range, 2)}")
print(f"Estimated Velocity (m/s): {round(det_velocity, 2)}")
print(f"Estimated Angle (deg): {round(det_angle, 2)})")
