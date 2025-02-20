"""
Interferometry Data Analysis v1.0

Lukas Kostal, 20.10.2022, ICL
"""

import numpy as np
import scipy as sp
import scipy.fft as sft
from matplotlib import pyplot as plt

# filename of dataset to analyze
filename = 'Data'

n_step = 1e6
v_step = 1e3
f_samp = 500

# number of meters moved per 1 micro-step
m_per_step = 3.81e-11

# load the data
data = np.loadtxt(r'/Users/mingzeyan/Desktop/Data1.txt', delimiter=' ', unpack=True)

# slice data to extract position and amplitude
x_arr = data[5,:]
y_arr = data[1,:]

# convert the position from steps to meters
x_arr = x_arr * m_per_step

# determine the number of samples
n_samp = len(x_arr)

# calculate the average distance between samples in m
m_per_samp = np.mean(np.diff(x_arr))

"""
should there be a factor of 2 ??
"""
t_per_samp = m_per_samp

# preform a discrete fourier transform for a real input
x_fft = sft.rfftfreq(n_samp)
y_fft = sft.rfft(y_arr)

# take the absolute value of the amplitude
y_fft = np.abs(y_fft)

# shift the spectrum to start at 0 frequency
xx_fft = x_fft[:int(len(x_fft)/2)]
yy_fft = y_fft[:int(len(y_fft)/2)]

# convert spectrum from oscillations per sample to meters per oscillation
xx_fft = t_per_samp / xx_fft

# calculate log for logarithmic spectrum
yy_fft_log = np.log(yy_fft)

# plot amplitude against displacement
plt.figure(1)
plt.plot(x_arr - 3e-5, y_arr)
plt.xlabel('Displacement from null point (m)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Amplitude against Mirror Displacement')

# plot FFT against spatial frequency
plt.figure(2)
plt.plot(x_fft, y_fft)

# plot spectral amplitude against wavelength
plt.figure(3)
plt.plot(xx_fft, yy_fft)
plt.xlim(1e-7,9e-7)
plt.xlabel('wavelength (m)')
plt.ylabel('amplitude (a.u.)')
plt.title('Reconstructed Spectrum')
plt.savefig('/Users/mingzeyan/Desktop/00', dpi=300)

# plot log of spectral amplitude against wavelength
plt.figure(4)
plt.plot(xx_fft, yy_fft_log)
plt.xlabel('wavelength (m)')
plt.ylabel('log of spectral amplitude (a.u.)')
plt.title('Log of Reconstructed Spectrum')
#plt.savefig('/Users/mingzeyan/Desktop/11', dpi=300)
plt.show()




