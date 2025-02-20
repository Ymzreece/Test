# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:44:00 2022
This is the test field of the simulation.
@author: Cocteau
"""
import numpy as np
#%%
"""
Task 2 Testing
"""
from Collision import Ball
A=Ball()
B=Ball(r=[4,0],v=[-1,0])
print(A.collide(B))
#%%
"""
Task 3,Task4 Testing
""" 
#use %matplotlib auto

from Collision import Simulation
A=Ball(1,1,[-5,0],[3,1])
c=Simulation(A)
print(c.next_collision())
c.run(30)
#%%
"""
Testing Balls Generator
"""
from Collision import balls_generator
A=balls_generator(10,1,1,5)   
print(A[0]._r,A[0]._v,A[9]._r,A[9]._v) 
#%%
"""
Task 7 Testing
"""
from Collision import Simulation
balls=balls_generator(40,1,1,5)  
c=Simulation(balls)
c.run(10)
#%%%

















