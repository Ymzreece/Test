#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:31:10 2023

@author: mingzeyan
"""

import matplotlib.pyplot as plt
import numpy as np
from math import atan2
from scipy.signal import find_peaks

#%%
#Run the FFT
time,amp, amp2= np.loadtxt('/Users/mingzeyan/Downloads/OneDrive_1_02-02-2023/1K500MS.CSV',skiprows=1,delimiter=',',unpack=True)
t = time
square_wave = amp

# Perform the FFT
fft = np.fft.fft(square_wave)
absfft = np.abs(fft)
absfft = list(absfft)
maxvalue = max(absfft)
max_index = (absfft.index(maxvalue))
sampling_rate = 1/(t[1]-t[0])
# Get the frequency axis
frequency_axis = np.fft.fftfreq(square_wave.size, d=1/sampling_rate)
print(fft[max_index].real)
peaks, _ = find_peaks(absfft[:int(len(t)/2)])

half = (np.abs(fft)[:int(len(t)/2)])
peaksy = sorted((half[peaks]))[-10:]
half = list(np.abs(fft)[:int(len(t)/2)])
print(peaksy)
peaks_in = []
for i in peaksy:
    peaks_in.append(half.index(i))
# Plot the original signal and its FFT
plt.subplot(2,1,1)
plt.plot(t, square_wave)
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Amplitude',fontsize=8)

plt.subplot(2,1,2)
plt.plot(frequency_axis, np.abs(fft))
plt.plot(frequency_axis[peaks_in], np.abs(fft)[peaks_in],'ro')
plt.xlabel('Frequency (Hz)',fontsize=8)
plt.ylabel('Amplitude',fontsize=8)
# peaks, _ = find_peaks(np.abs(fft))
# peaks_sorted = sorted(peaks, key=lambda frequency_axis: np.abs(fft)[frequency_axis], reverse=True)
# for i in peaks_sorted:
#     plt.plot(frequency_axis[i], peaks_sorted[i], "ro")
plt.show()
# plt.savefig('/Users/mingzeyan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Waves Year 2 Lab/Post Lab 3/1K FFT', dpi=1000)

#%%
fnames=["1K","5K"]
frequency=[1000,5000]
for i in range (1,15):
    a=10*i
    fnames.append(f"{a}K")
    frequency.append(10000*i)
print(fnames,frequency)
#%%
#Plot the phase difference against frequency
def phasediff(fnamelist,num_peaks):
    
    for i in fnamelist:
        phase_diff=[]
        time, amp1, amp2 = np.loadtxt(f'/Users/mingzeyan/Downloads/OneDrive_1_02-02-2023/{i}.CSV',skiprows=1,delimiter=',',unpack=True)
        t = time
        
        square_wave_1 = amp1
        square_wave_2 = amp2
        
        #Complex valued
        fft1 = np.fft.fft(square_wave_1)
        fft2 = np.fft.fft(square_wave_2)
        absfft1 = np.abs(fft1)
        absfft2 = np.abs(fft2)
        sampling_rate = 1/(t[1]-t[0])
        
        frequency_axis_1 = np.fft.fftfreq(square_wave_1.size, d=1/sampling_rate)
        frequency_axis_2 = np.fft.fftfreq(square_wave_2.size, d=1/sampling_rate)
        
        len_1 = int(len(frequency_axis_1)/2)
        len_2 = int(len(frequency_axis_2)/2)
        
        fft1_half = fft1[:len_1]
        fft2_half = fft2[:len_2]
        frequency_axis_1_half = frequency_axis_1[:len_1]
        frequency_axis_2_half = frequency_axis_2[:len_1]

        absfft1_half = (np.abs(fft1)[:len_1])
        absfft2_half = (np.abs(fft2)[:len_2])
        peaks_1, _ = find_peaks(absfft1_half)
        peaks_2, _ = find_peaks(absfft2_half)

        peaksy_1 = sorted((absfft1_half[peaks_1]))[-num_peaks:]
        peaksy_2 = sorted((absfft2_half[peaks_2]))[-num_peaks:]
        print(peaksy_1)
        absfft1_half = list(np.abs(fft1)[:len_1])
        absfft2_half = list(np.abs(fft2)[:len_2])


        
        peaks_index_1 = []
        peaks_index_2 = []
        
        for i in peaksy_1:
            peaks_index_1.append(absfft1_half.index(i))
        for i in peaksy_2:
            peaks_index_2.append(absfft2_half.index(i))
        print(peaks_index_1)
        peaks_x_needed_1 = []
        peaks_x_needed_2 = []
        
        for i in peaks_index_1:
            peaks_x_needed_1.append(frequency_axis_1_half[i])
        for i in peaks_index_2:
            peaks_x_needed_2.append(frequency_axis_2_half[i])

        peaks_x_needed_1 = np.array(peaks_x_needed_1)
        peaks_x_needed_2 = np.array(peaks_x_needed_2)
        
        peaks_x_needed = (peaks_x_needed_1 + peaks_x_needed_2)/2
        peaks_needed_1 = []
        peaks_needed_2 = []
        
        for i in peaks_index_1:
            peaks_needed_1.append(fft1_half[i])
        for i in peaks_index_2:
            peaks_needed_2.append(fft2_half[i])

        reals_1 = []
        reals_2 = []
        im_1 = []
        im_2 = []
        
        for i in peaks_needed_1:
            reals_1.append(i.real)
            im_1.append(i.imag)
        for i in peaks_needed_2:
            reals_2.append(i.real)
            im_2.append(i.imag)
        
        reals_1 = np.array(reals_1)
        reals_2 = np.array(reals_2)
        im_1 = np.array(im_1)
        im_2 = np.array(im_2)
        
        phase_1 = []
        phase_2 = []
        for i in range(num_peaks):
            phase_1.append(atan2(im_1[i],reals_1[i]))
            phase_2.append(atan2(im_2[i],reals_2[i]))
        phase_1 = np.array(phase_1)
        phase_2 = np.array(phase_2)
        phasedif = phase_1 - phase_2
        phase_diff.append(phasedif)
        plt.subplot(3,1,1)
        plt.plot(peaks_x_needed,phase_diff[0],'x')
        plt.xlabel('Frequency, Hz')
        plt.ylabel('Phase Difference')
        plt.title('Phase Lag')
        plt.subplot(3,1,2)
        plt.plot(frequency_axis, absfft1)
        plt.plot(peaks_x_needed_1, peaksy_1,'ro')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Section 1')
        plt.subplot(3,1,3)
        plt.plot(frequency_axis, absfft2)
        plt.plot(peaks_x_needed_2, peaksy_2,'ro')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Section 40')
        
    return peaks_x_needed,phase_diff

phasediff(["1K"],20)
# phasediff(["5K"],4)
# phasediff(["10K"],4)
# phasediff(["20K"],3)
# phasediff(["30K"],2)
# phasediff(["40K"],2)
# phasediff(["50K"],1)
# phasediff(["60K"],1)
# phasediff(["70K"],1)
# phasediff(["80K"],1)
# phasediff(["90K"],1)
# phasediff(["100K"],1)
# phasediff(["110K"],1)
# phasediff(["120K"],1)
# phasediff(["130K"],1)
# phasediff(["140K"],1)


    
    
    
    
    
    
    
    
    
    
    
    

