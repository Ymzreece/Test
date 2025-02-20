# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:34:36 2022

@author: Cocteau
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
#%%
"""
Task 2
"""
class Ball:
    """
    This is the simulation of movments and collisions between balls.
    """

    def __init__(self, m=1.0,R=1.0, r=[0,0],v=[0,0]):
        
        self.m=m
        self.R=R
        self._r=np.asarray(r)
        self._v=np.asarray(v)
        if len(self._r)!=2:
            raise Exception("Ball parameter r has incorrect size")
        if len(self._v)!=2:
            raise Exception("Ball parameter v has incorrect size")
    def __repr__(self):
        return f"(m = {self.m:.1f},R = {self.R:.1f},r={self._r},v={self._v})"
    def pos(self):
        return self._r
    def vel(self):
        return self._v
    def move(self, dt):
        self._r=self._r+dt*self._v
    def time_to_collision(self,other):
        """
        Here we use (1) in the case of + sign.
        """
        relative_r=self._r-other._r
        #print("relative_r",relative_r)
        relative_v=other._v-self._v
        relative_speed=np.dot(relative_v,relative_r)/np.linalg.norm(relative_r)
        relative_R=self.R+other.R
        if np.linalg.norm(relative_r)<=relative_R:
            t=0
            """
            This is caused by numerical error
            """
        if relative_speed >0:
            dt=(np.linalg.norm(relative_r)-relative_R)/relative_speed
        if relative_speed <=0:
            "the balls never collide"
            dt=1000
        return dt
    def collide(self, other):
        relative_v=other._v-self._v
        self.v=(2*other.m*other._v+self._v*(self.m-other.m))/2*self.m
        other.v=self._v-relative_v
        return(self.v,other.v)
    def collision_time_with_wall(self):
        #vt = self.Ball._v
        coeff=[(np.linalg.norm(self._v))**2,2*np.dot(self._r,self._v),(np.linalg.norm(self._r))**2-(self.R-10)**2]
        print("root",np.roots(coeff))
        if len(np.roots(coeff))==0:
            t=1000
            return t
        else:
            if np.roots(coeff)[0]>1e-6:
                t=np.roots(coeff)[0]
                return t
            else:
                t=np.roots(coeff)[1]
                return t
    def collide_with_wall(self):
        v_radial=np.dot(self._r,self._v)*self._r/(np.linalg.norm(self._r))**2
        v_tangent=self._v-v_radial
        self._v=v_tangent-v_radial
        return self._v
    def get_patch(self):
        self.patch.center = self._r
        #print(self.patch.center)
        return self.patch
        

     
#%%%
"Generating Random Balls"
import random as rn
def balls_generator(n,m,R,vmax):
    i=1
    balls=[]
    balls.append(Ball(m,R,[0,0],[0,0]))
    while i<n:
        x=rn.uniform(-9,9)
        y=rn.uniform(-np.sqrt(81-x**2),np.sqrt(81-x**2))
        vx=rn.uniform(-vmax,vmax)
        vy=rn.uniform(-vmax,vmax)
        #print(x,y,vx,vy)
        new_ball=Ball(m,R,[x,y],[vx,vy])
        j=0
        while j<i:
            if (balls[j]._r[0]-x)**2+(balls[j]._r[1]-y)**2>=(2*R)**2:
               j+=1
               #print(j,"j")
            else:
                break
            if j==i:
                balls.append(new_ball)
                i+=1
               # print(i)
                break
    return balls  
"""
If this code is continously running,
this implies that the number of generating balls are too large for the contaniner to handel.
Please reduce the radius of the ball
"""

         
 #%%%        
'''
Task 3,Task 7
'''
class Simulation(Ball):
     """
     #This is the simulation of movments and collisions between a ball and a wall.
     """
     n=0
     p=0
     m=0
     dt=999
     def __init__(self,theBalls,R_container=10.0): 
        if(isinstance(theBalls,list)):
            self.myBall=theBalls
        else:
            self.myBall=[theBalls]
        self.R_container=R_container
     def __repr__(self):
        return f"(m = {self.myBall.m:.1f},R = {self.myBall.R:.1f},R_container = {self.R_container:.1f}r={self.myBall._r},v={self.myBall._v})" 
     def True_collision(self):
        self.dt=999
        for i in np.arange(0,len(self.myBall)):
            t_wall=self.myBall[i].collision_time_with_wall()
            for j in np.arange(i+1,len(self.myBall)):
                collision_time=self.myBall[i].time_to_collision(self.myBall[j])
                #print("dt",self.dt)
                if collision_time<=self.dt:
                    self.dt=collision_time
                    self.p=0
                    self.n=i
                    self.m=j
            if t_wall<=self.dt:
                self.dt=t_wall
                #print("dt2",self.dt)
                self.p=1
                self.n=i
                self.m=0      
     def next_collision(self):
         self.True_collision()
         dt=self.dt
         print("true dt",dt)
         for i in range(len(self.myBall)):
             self.myBall[i].move(dt)
         if self.p==0:
             self.myBall[self.n].collide(self.myBall[self.m])
         if self.p==1:
             self.myBall[self.n]._v=Simulation.collide_with_wall(self.myBall[self.n])
     def run(self, num_frames, animate=False):
         if animate:
             f = pl.figure()
             ax = pl.axes(xlim=(self.R_container, self.R_container), ylim=(self.R_container, self.R_container))
             ax.set_aspect( 1 )
             ax.add_artist(pl.Circle([0., 0.], self.R_container, ec='g', fill=False, ls='solid'))
             for i in np.arange(0,len(self.myBall)):
                 ax.add_patch(self.myBall[i].get_patch())
            #ax.add_patch(self.ball1.get_patch())
         for frame in range(num_frames):
             self.next_collision()
             
             self.vel = []
             self.dis1 = []
             self.dis2 =[]
             for i in np.arange(1,len(self.myBall)):
                 self.vel.append((self.myBall[i]._v))
                 for j in np.arange(i+1,len(self.myBall)):
                     dis1 = np.linalg.norm(self.myBall[i]._r-self.myBall[j]._r)
                     
                     self.dis1.append(dis1)
                 dis2 = np.linalg.norm(self.myBall[i]._r)
                 self.dis2.append(dis2)
             # print('The total Kinetic Energy is',0.5*1*sum(self.vel), 'J.')
             
             if animate:
                 pl.pause(1)
         if animate:
             pl.show()    
         plt.hist(self.dis1,100)
         plt.title("distance between balls")
         plt.grid()
         plt.show()
         plt.hist(self.dis2,100)
         plt.title("positions of balls")
         plt.grid()
         plt.show()
#%%%       
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
        
           

