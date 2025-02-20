#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 02:27:25 2023

@author: mingzeyan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

preg1,gluc1,bp1,skinthic1,insulin1,BMI1,dpf1,age1,outcome1 = np.loadtxt('/Users/mingzeyan/Desktop/md_1.txt',delimiter='\t',unpack=True)
# preg0,gluc0,bp0,skinthic0,insulin0,BMI0,dpf0,age0,outcome0 = np.loadtxt('/Users/mingzeyan/Desktop/md_0.txt',delimiter='\t',unpack=True)
# pregd0,glucd0,bpd0,skinthicd0,insulind0,BMId0,dpfd0,aged0,outcomed0 = np.loadtxt('/Users/mingzeyan/Desktop/md_d_0.txt',delimiter='\t',unpack=True)
pregd1,glucd1,bpd1,skinthicd1,insulind1,BMId1,dpfd1,aged1,outcomed1 = np.loadtxt('/Users/mingzeyan/Desktop/md_1.txt',delimiter='\t',unpack=True)
diabetes = pd.read_csv('/Users/mingzeyan/Desktop/modeling_diabetes.csv')
#%%
# diabetes=diabetes[(diabetes.BloodPressure != 0)&(diabetes.BMI != 0)&(diabetes.Glucose != 0)&(diabetes.SkinThickness != 0)]
diabetes.groupby('Outcome').hist(figsize=(9,9))
#%%
plt.figure(dpi = 200,figsize=(6,5))
plt.tick_params(labelsize=4)
P = sns.heatmap(diabetes.corr(),annot=True)
#%%
plt.hist(preg1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('Times of pregnancy')
#%%
plt.hist(pregd1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('Times of pregnancy')
#%%
plt.hist(gluc1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('Glucose')
#%%
plt.hist(glucd1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('Glucose')
#%%
plt.hist(bp1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('Blood Pressure')
#%%
plt.hist(bpd1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('Blood Pressure')
#%%
plt.hist(skinthic1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('skin thickness')
#%%
plt.hist(skinthicd1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('skin thickness')
#%%
plt.hist(insulin1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('insulin')
#%%
plt.hist(insulind1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('insulin')
#%%
plt.hist(BMI1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('BMI')
#%%
plt.hist(BMId1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('BMI')
#%%
plt.hist(dpf1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('Diabetes Pedigree Function')
#%%
plt.hist(dpfd1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('Diabetes Pedigree Function')
#%%
plt.hist(age1,density=True,bins=10)
plt.title('Percentage of diabetes patients (with outliers)')
plt.ylabel('Percentage')
plt.xlabel('age')
#%%
plt.hist(aged1,density=True,bins=10)
plt.title('Percentage of diabetes patients (without outliers)')
plt.ylabel('Percentage')
plt.xlabel('age')















