#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:21:48 2022

@author: mingzeyan
"""

import numpy as np
import pylab as pl

#task 2
class Ball:
    def __init__(self,m,r,p,u):
        self.mass = m
        self.radius = r
        self.position = np.array(p)
        self.velocity = np.array(u)
        
        if self.mass < 10**12:
            self.patch = pl.Circle(self.position,self.radius,fc='r')
            
        if self.mass == 10**12:
            self.patch = pl.Circle(self.position,self.radius,ec = 'b', fill = False)
    def pos(self):
        print(self.position)
        return self.position
    def vel(self):
        return self.velocity
    def move(self, dt):
        self.position = self.position + self.velocity * dt
        self.patch.center = self.position
        return self.position
    
    def time_to_collision(self,other):
        # if self.mass > 1000:
        #     self.radius = -1 * self.radius
            
        # if other.mass > 1000:
        #     other.radius = -1 * other.radius
            
        r_p = self.position - other.position
        r_u = self.velocity - other.velocity
        pu = np.dot(r_p,r_u)
        pp = np.dot(r_p,r_p)
        uu = np.dot(r_u,r_u)
        # if self.mass > 1000 or other.mass>1000:
        #     rr = (self.radius - other.radius)**2
        # if self.mass < 1000 and other.mass<1000:
            # rr = (self.radius + other.radius)**2
        rr = (self.radius + other.radius)**2
        if 4*pu**2 - 4*uu*(pp-rr) < 0:
            return ('There is no collision')
        else:
            t_1 = (-2*pu - np.sqrt(4*pu**2 - 4*uu*(pp-rr)))/(2*uu)
            t_2 = (-2*pu + np.sqrt(4*pu**2 - 4*uu*(pp-rr)))/(2*uu)
            if t_1  > 0 and t_2 > 0:
                t_d = t_1 - t_2
                if t_d > 0:
                    return t_2
                else:
                    return t_1
                if t_1 < 0 and t_2 > 0:
                    return t_2
                if t_1 < 0 and t_2 < 0:
                    return ('There is no collision')
        if other.mass == 10**12:
            t = (-1*np.dot(r_p,self.velocity)+np.sqrt((np.dot(r_p,self.velocity)**2)-np.dot(self.velocity,self.velocity)*(pp-(self.radius-other.radius)**2)))/np.dot(self.velocity,self.velocity)
            return t
        if self.mass == 10**12:
            t = (-1*np.dot(-r_p,other.velocity)+np.sqrt((np.dot(-r_p,other.velocity)**2)-np.dot(other.velocity,other.velocity)*(pp-(other.radius-self.radius)**2)))/np.dot(other.velocity,other.velocity)
            return t
    def collide(self,other):
        rdif = self.position - other.position
        # rval = np.sqrt(np.dot(rdif,rdif))
        # rhat = rdif/rval
        # u1_h = np.dot(self.velocity,-rhat)
        # u2_h = np.dot(other.velocity,rhat)
        # v1_h = (self.mass-other.mass)*u1_h/(self.mass+other.mass)+2*other.mass*u2_h/(self.mass+other.mass)
        # v2_h = 2*self.mass*u1_h/(self.mass+other.mass)-(self.mass-other.mass)*u2_h/(self.mass+other.mass)
        # u1_mod = np.sqrt(self.velocity[0]**2+self.velocity[1]**2)
        # u2_mod = np.sqrt(other.velocity[0]**2+other.velocity[1]**2)
        # v1_v = np.sqrt(u1_mod**2 - u1_h**2)
        # v2_v = np.sqrt(u2_mod**2 - u2_h**2)
        # v1 = np.array([v1_h,v1_v])
        # v2 = np.array([v2_h,v2_v])
        mdif = self.mass - other.mass
        msum = self.mass + other.mass
        v1 = self.velocity - 2*other.mass*np.dot(self.velocity-other.velocity,rdif)*rdif/(msum*np.linalg.norm(rdif)**2)
        
        v2 = other.velocity - 2*self.mass*np.dot(other.velocity-self.velocity,-1*rdif)*-1*rdif/(msum*np.linalg.norm(-rdif)**2)
        self.velocity = v1
        other.velocity = v2
        if self.mass == 10**12:
            
            return other.velocity
        if other.mass == 10**12:
            
            return self.velocity
        if self.mass < 10**12 and other.mass < 10**12:
            return self.velocity, other.velocity
        
    def get_patch(self):
        self.patch.center = self.position
        print(self.patch.center)
        print(self.patch)
        return self.patch
    def move_patch(self):
        self.patch.center = self.position
        print(self.patch.center)
        return self.patch.center


#task 3
class Simulation(Ball):
    def __init__(self,particles):
        self.container = particles[1]
        self.ball = particles[0] 
        #self.ball1 = particles[2]
        
        self.particles = particles
    def next_collision(self):
        # for i in np.arange (0,len(self.balls)):
        #     for j in np.arange(i+1,len(self.balls)):
        #         next_collision_time = self.balls[i].time_to_collision(self.balls[j])
        #         if next_collision_time < min_collision_time:
        #             min_collision_time = next_collision_time
        # return min_collision_time
        
        # min_collision_time = 1000000000000
        # for i in range(len(self.particles)):
        #     for j in np.arange(i+1,len(self.particles)):
        #         next_collision_time=self.balls[i].time_to_collision(self.balls[j])
        #         if next_collision_time == None:
        #             return ('There is no collision anyore')
        #         if next_collision_time < min_collision_time:
        #             min_collision_time = next_collision_time
        # return min_collision_time
        time = Ball.time_to_collision(self.ball,self.container)
        # time = int(time)
        # time = float(time)
        self.ball.position = self.ball.position + self.ball.velocity * time
        self.ball.velocity = self.ball.collide(self.container)
        self.container.position = np.array([0.,0.])
        self.container.velocity = np.array([0.,0.])
        
        #time1 = Ball.time_to_collision(self.ball1,self.ball)
        #time2 = Ball.time_to_collision(self.ball1,self.container)
        #TIME = np.array([time0,time1,time2])
       # mintime = min(TIME)
        #print(time0,time1,time2,mintime)
        # if mintime == time0:
        #     self.ball.move(time0)
        #     self.ball.collide(self.container)
        #     return time0, self.ball.position, self.ball.velocity
        # if mintime == time1:
        #     self.ball.move(time1)
        #     self.ball1.move(time1)
        #     self.ball.collide(self.ball1)
        #     return time1, self.ball.position, self.ball.velocity,self.ball1.velocity
        # if mintime == time2:
        #     self.ball1.move(time2)
        #     self.ball1.collide(self.container)
        return time, self.ball.position, self.ball.velocity, self.container.position, self.container.velocity
    def run(self, num_frames, animate=False):
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.set_aspect( 1 )
            ax.add_artist(self.container.get_patch())
            ax.add_patch(self.ball.get_patch())
           #ax.add_patch(self.ball1.get_patch())
        for frame in range(num_frames):
            self.next_collision()
            if animate:
                self.ball.patch.center = (self.ball.move_patch())
                pl.pause(1)
        if animate:
            pl.show()

Container = Ball(10**12,10,[0,0],[0,0])
BALL_1 = Ball(1,1,[-5,0],[1,2])

test = Ball(10**12,10,[0,0],[0,0])
test1 = Ball(1,1,[2,0],[-2,-2])

particles = (BALL_1,Container)

Test = Simulation(particles)







    
                
        
        
        
        
        
        
