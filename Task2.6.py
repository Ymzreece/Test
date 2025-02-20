#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:15:17 2023

@author: alanli
"""
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

#%%
#4mina
time,tem = np.loadtxt('/Users/mingzeyan/Downloads/thermal_8min.txt',skiprows = 3 ,unpack = True)
time=time[0:600]
tem = tem[0:600]
square = [100]*300+[0]*300

def cos4min(t,n):
 return np.cos(2*np.pi*n*t/600)
def sin4min(t,n):
 return np.sin(2*np.pi*n*t/600)

def nuint(x,y):
    area=0
    for i in range (0,len(x)-1):
        if y[i+1]>0:
            area=area+(np.abs(y[i])+np.abs(y[i+1]))*0.5*(x[i+1]-x[i])
        if y[i+1]<0:
            area=area-(np.abs(y[i])+np.abs(y[i+1]))*0.5*(x[i+1]-x[i])
    return area

def harmonic(t, a, b, n):
    return np.sqrt(a**2+b**2)*np.sin(np.pi*n/300*t+np.arctan2(a,b))

def intlist(n):
    return [i for i in range(1, n+1)]


def harmplot (time, numberlist, outorin):
    plt.plot(time, outorin, label='Waveform')
    a0=(1/300)*nuint(time,outorin)
    for i in numberlist:
        htotal=np.array([0]*600)
        for n in intlist(i):
            if n==1:
                htotal=np.array([a0/2]*600)
            cos=[]
            sin=[]
            for t in time:
                cos.append(cos4min(t,n))
                sin.append(sin4min(t,n))
            cos=np.array(cos)
            sin=np.array(sin)
    
            producta=outorin*cos
            productb=outorin*sin
            a=(1/300)*nuint(time,producta)
            b=(1/300)*nuint(time,productb)
            
            h=[]
            for t in time:
                h.append(harmonic(t, a, b, n))
            htotal=htotal+np.array(h)
        plt.plot(time, htotal, label=f"sum of {i} harmonics")
        plt.legend()
    plt.xlabel('time/ds')
    plt.ylabel('temperature/ $^oC$')
    plt.title('4 minute a Fourier Analysis')
    if sum(outorin) == sum(tem):
        plt.savefig("2.6a.png", dpi=1000)
    if sum(outorin) == sum(square):
        plt.savefig("2.6b.png", dpi=1000)
    
    
def betaphi (time, numberlist, outorin):
    for n in numberlist:
        cos=[]
        sin=[]
        for t in time:
            cos.append(cos4min(t,n))
            sin.append(sin4min(t,n))
        cos=np.array(cos)
        sin=np.array(sin)

        producta=outorin*cos
        productb=outorin*sin
        a=(1/300)*nuint(time,producta)
        b=(1/300)*nuint(time,productb)
        beta=np.sqrt(a**2+b**2)
        deltaphi=-np.arctan2(a,b)
        if deltaphi < 0:
            deltaphi = deltaphi + 2*np.pi
        else: 
            deltaphi = deltaphi + np.pi
        # print(f"a_{n} is {a:.3f}")
        # print(f"b_{n} is {b:.3f}")
        # print(f"β_{n} is {beta:.3f}")
        # print(f"Δφ_{n} is {deltaphi:.3f}")
        # print(" ")
    return a,b,beta,deltaphi
#%%
def D_pl(n):
    w=[]
    phaselag=[]
    for i in n:
        w.append(2*np.pi*i/600)
        phaselag.append(betaphi(time, [i], tem)[3])
    w=np.array(w)
    phaselag=np.array(phaselag)
    
    print('Phaselag=',phaselag)
    return (w*60.0625)/(2*phaselag**2)
#%%
def D_tf(n):
    w=[]
    gamma=[]
    for i in n:
        w.append(2*np.pi*i/600)
        gamma.append(betaphi(time, [i], tem)[2]/betaphi(time,[i], square)[2])
    w=np.array(w)
    gamma=np.array(gamma)

    print('gamma = ',gamma)
    return (w*60.0625)/(2*(np.log(gamma))**2)

#%%

def errordtf(n):
    error=[]
    a=[]
    b=[]
    for i in n:
        a.append(betaphi(time, [i], tem)[0])
        b.append(betaphi(time, [i], tem)[1])
        error.append(np.sqrt(((betaphi(time, [i], tem)[0])**2*3.5355e-5)**2 + ((betaphi(time, [i], tem)[1])**2*3.5355e-5)**2))
    percenterror=np.array(error)/(np.array(a)**2+np.array(b)**2)
    return D_tf([1,3,5,7,9,11,13,15,17])*np.sqrt(percenterror**2+0.00365**2)

plt.errorbar([1,3,5,7,9,11,13,15,17],D_pl([1,3,5,7,9,11,13,15,17]),yerr = D_pl([1,3,5,7,9,11,13,15,17])*0.00365,fmt='.',label="$D_{PL}$")
plt.errorbar([1,3,5,7,9,11,13,15,17], D_tf([1,3,5,7,9,11,13,15,17]), yerr = errordtf([1,3,5,7,9,11,13,15,17]), fmt=".", label="$D_{TF}$")
plt.legend()
plt.xlabel('Harmonic number')
plt.ylabel('Diffusivity, $mm^2ds^{-1}$')
plt.title('Diffusivity for different harmonics of 1 min a data')
# plt.savefig('Task2.6b2.png', dpi=1000)

