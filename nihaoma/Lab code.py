#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:14:13 2023

@author: mingzeyan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#task 2.3(a)
time,tem_a = np.loadtxt('/Users/mingzeyan/Desktop/thermal_4min_a.txt',skiprows = 3 ,unpack = True)
# plt.plot(time,tem_a,label='inner temperature')
tem_square_wave = [100]*1200+[0]*1200+[100]*1200+[0]*1200+[100]*1200+[0]*1200+[100]*1200+[0]*1200
tem_square_wave = np.array(tem_square_wave)
plt.plot(time,tem_square_wave, label='outer temperature')
plt.legend()
plt.xlabel('time/ds')
plt.ylabel('temperature/ $^oC$')
#plt.savefig("2.3plot(a).png", dpi=1000)
#%%
#Task 2.3(b)
# plt.plot(time,tem_a,label='Inner temperature')
tem_n1_wave = ([100]*1200+[0]*1200)*4
tem_n1_wave = 50+63.662*np.sin(2*np.pi*time/2400)
plt.plot(time,tem_n1_wave, label='Foundamental Harmonic')
plt.legend()
plt.xlabel('time/ds')
plt.ylabel('temperature/ $^oC$')
#plt.savefig("2.3plot(b).png", dpi=1000)
#print((max(tem_a)-min(tem_a))/2)




#%%
def sine_fit(x, amp, angfre, lag, shift):
    return amp * np.sin(angfre * x + lag) + shift
popt, pcov = curve_fit(sine_fit, time, tem_a, p0=[10,np.pi/1200, np.pi, 50])
plt.plot(time,tem_a,'x')
plt.plot(popt[0]*np.sin(popt[1]*time+popt[2])+popt[3])
error=(np.sqrt(np.diag(pcov)))
print('amplitude =', popt[0],'+/-',error[0])
print('lag =', popt[2],'+/-',error[2])
print('average temperature =', popt[3],'+/-',error[3])





















