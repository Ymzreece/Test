import numpy as np
import pylab as pl
import random as rn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def balls_generator(num_balls,mass,radius,sigma):
    '''
    This function is independent from any classes. This is used to generate
    balls at random position and random velocity. It will also automatically
    generate a container with mass 10e30, radius 10. The mean velocity of all
    balls is equal to 0.

    Parameters
    ----------
    num_balls : int
        The number of balls it will generate.
    mass : float
        The mass of a ball
    radius : float
        The radius of a ball
    sigma : float
        The standard deviation of the velocity distribution. It determines the
        range of the velocity.

    Returns
    -------
    balls : TYPE
        DESCRIPTION.

    '''
    i=1
    balls=[Ball(10**30,10,[0,0],[0,0])]
    balls.append(Ball(mass,radius,[0,0],[0,0]))
    while i<num_balls:
        x=rn.uniform(-9,9)
        y=rn.uniform(-np.sqrt(81-x**2),np.sqrt(81-x**2))
        v = np.random.normal(0,sigma,2)
        #print(x,y,vx,vy)
        new_ball=Ball(mass,radius,[x,y],v)
        j=0
        while j<i:
            if (balls[j].position[0]-x)**2+(balls[j].position[1]-y)**2>=(2*radius)**2:
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

def func_gau(x,a,b,c,d):
    '''
    You do not need to consider this function. This function is independent
    from any classes. It is used to curve_fit function
    '''
    y = a*x*np.exp(-0.5*b*x**2/(c*d))
    return y

class Ball:
    def __init__(self,m,r,p,u):
        '''
        

        Parameters
        ----------
        m : int
            this is the mass of a ball or container
        r : int
            this is the radius of a ball or container
        p : array
            this is the initial position of a ball or container
        u : array
            this is the initial velocity of a ball

        Returns
        -------
        None.

        '''
        self.mass = m
        self.radius = r
        self.position = np.array(p)
        self.velocity = np.array(u)
        
        if self.mass < 10**30:
            self.patch = pl.Circle(self.position,self.radius,fc='r')
        if self.mass == 10**30:
            self.patch = pl.Circle(self.position,self.radius,ec = 'b', fill = False)

    def pos(self):
        return self.position

    def vel(self):
        return self.velocity

    def move(self, dt):
        '''
        
        Parameters
        ----------
        dt : float
            dt is time the ball moved

        Returns
        -------
        TYPE array
            position of ball is returned, so the position will be undated after moving

        '''
        self.position = self.position + self.velocity * dt
        self.patch.center = self.position
        return self.position
    
    def time_to_collision(self,other):
        '''
        This function is used to calculate the time needed to collide between
        two balls.

        Parameters
        ----------
        self : Ball
            One of the balls colliding
        other : Ball
            The other ball collisding

        Returns
        -------
        TYPE float
            it returns the time used between two balls to the next collision

        '''
        r_p = self.position - other.position
        r_u = self.velocity - other.velocity
        pu = np.dot(r_p,r_u)
        pp = np.dot(r_p,r_p)
        uu = np.dot(r_u,r_u)
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
        if other.mass == 10**30:
            t = (-1*np.dot(r_p,self.velocity)+np.sqrt((np.dot(r_p,self.velocity)**2)-np.dot(self.velocity,self.velocity)*(pp-(self.radius-other.radius)**2)))/np.dot(self.velocity,self.velocity)
            return t
        if self.mass == 10**30:
            t = (-1*np.dot(-r_p,other.velocity)+np.sqrt((np.dot(-r_p,other.velocity)**2)-np.dot(other.velocity,other.velocity)*(pp-(other.radius-self.radius)**2)))/np.dot(other.velocity,other.velocity)
            return t
        
    def collide(self,other):
        '''
        This function is used to calculate the velocities of balls after each
        collision.

        Parameters
        ----------
        self : Ball
            One of the balls colliding.
        other : Ball
            The other ball colliding.

        Returns
        -------
        TYPE array
            this returns the velocity of one of the balls after collision.
        TYPE array
            this returns the velocity of the other ball after collision.

        '''
        rdif = self.position - other.position
        msum = self.mass + other.mass
        v1 = self.velocity - 2*other.mass*np.dot(self.velocity-other.velocity,rdif)*rdif/(msum*np.linalg.norm(rdif)**2)
        v2 = other.velocity - 2*self.mass*np.dot(other.velocity-self.velocity,-1*rdif)*-1*rdif/(msum*np.linalg.norm(-rdif)**2)
        self.velocity = v1
        other.velocity = v2
        return self.velocity, other.velocity
    
    def new_collide(self,other):
        '''
        
        You do not need to consider this function. Because it is the same
        usage as 'collide' function, but it does not return both of velocities.
        Then the velocities are not updated after collision.
        
        '''
        rdif = self.position - other.position
        msum = self.mass + other.mass
        v1 = self.velocity - 2*other.mass*np.dot(self.velocity-other.velocity,rdif)*rdif/(msum*np.linalg.norm(rdif)**2)
        return v1
        
    def get_patch(self):
        '''
        
        You do not need to consider this function. This function is used to
        draw balls in the animation.
        
        '''
        self.patch.center = self.position
        return self.patch


#task 3
class Simulation(Ball):
    def __init__(self,particles):
        '''
        

        Parameters
        ----------
        particles : List of Balls
            This is a list of balls, which is consist of all information we
            need, mass, radius, position and velocity.

        Returns
        -------
        None.

        '''
        self.t = 0.0
        self.change_in_momentum = 0
        self.container = particles[0]        
        self.balls = particles
        
    def min_collision_time(self):
        '''
        You do not need to consider about this function, it will run
        automatically.

        Returns
        -------
        min_collision_time : float
            There are a lot of balls moving at the same time in the container.
            It returns the shortest time that the first collision happens.
        index_i : int
        index_j : int
            index_i and index_j determine which two balls in the container
            collide in the shortest time.

        '''
        min_collision_time = 1000000000000
        for i in np.arange (0,len(self.balls)):
            for j in np.arange(i+1,len(self.balls)):
                next_collision_time = self.balls[i].time_to_collision(self.balls[j])
                if next_collision_time is None:
                    next_collision_time = 1000
                if next_collision_time == 'There is no collision':
                    next_collision_time = 1000
                if next_collision_time < min_collision_time:
                    min_collision_time = next_collision_time
                    index_i = i
                    index_j = j
        return min_collision_time, index_i, index_j
    

    def next_collision(self):
        '''
        
        You do not need to consider about this function. This function returns
        exact the same variables as 'min_collision_time' function. But it
        updates the position and velocity of each ball after every collision.
        
        '''
        dt, index_i, index_j = self.min_collision_time()
        for ball in self.balls:
            ball.move(dt)
        self.balls[index_i].collide(self.balls[index_j])
        if index_i == 0:
            v_i = self.balls[index_i].velocity
            v_f = self.balls[index_i].new_collide(self.balls[index_j])
            momentum_change = self.balls[index_i].mass*np.linalg.norm(v_f-v_i)
            self.change_in_momentum = self.change_in_momentum + momentum_change
        if index_j == 0:
            v_i = self.balls[index_j].velocity
            v_f = self.balls[index_j].new_collide(self.balls[index_i])
            momentum_change = self.balls[index_j].mass*np.linalg.norm(v_f-v_i)
            self.change_in_momentum = self.change_in_momentum + momentum_change            
        return dt, index_i, index_j
    
    def run(self, num_frames, animate=True):
        '''
        This function shows the animation of balls collision.

        Parameters
        ----------
        num_frames : int
            How many frames it will perform.
        animate : True or False
            The default is True. You do not need to consider this one. Only if
            it is True, the animation will show up.

        Returns
        -------
        None.

        '''
        self.particles = self.balls[1:]
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-self.container.radius, self.container.radius), ylim=(-self.container.radius, self.container.radius))
            ax.set_aspect( 1 )
            ax.add_artist(self.container.get_patch())
            for i in np.arange(0,len(self.particles)):
                ax.add_patch(self.particles[i].get_patch())
        for frame in range(num_frames):
            self.next_collision()
            
            self.vel = []
            for i in np.arange(1,len(self.balls)):
                self.vel.append(np.linalg.norm(self.balls[i].velocity)**2)

            if animate:
                pl.pause(1)
        if animate:
            pl.show()
            
    def dis_int(self, num_frames):
        '''
        This function is used to generate the histogram of the distance between
        balls.

        Parameters
        ----------
        num_frames : int
            The histogram will show up after the number of frames you typed in.
            The more frames, the more accurate the histogram.

        Returns
        -------
        None.

        '''
        self.particles = self.balls[1:]
        self.dis = []
        for frame in range(num_frames):
            self.next_collision()
            for i in np.arange(1,len(self.balls)):
                for j in np.arange(i+1,len(self.balls)):
                    dist = np.linalg.norm(self.balls[i].position-self.balls[j].position)            
                    self.dis.append(dist)
        plt.hist(self.dis,100)
        plt.xlabel('Distance between balls (m)')
        plt.ylabel('Number of Measurements')
        plt.title('Distribution of distance between balls')
            
    def dis_cen(self, num_frames):
        '''
        This function is used to generate the histogram of the distance
        between balls and the centre.

        Parameters
        ----------
        num_frames : int
            The histogram will show up after the number of frames you typed in.
            The more frames, the more accurate the histogram.

        Returns
        -------
        None.

        '''
        self.particles = self.balls[1:]
        self.dis = []
        for frame in range(num_frames):
            self.next_collision()
            for i in np.arange(1,len(self.balls)):
                dist = np.linalg.norm(self.balls[i].position)
                self.dis.append(dist)
        plt.hist(self.dis,100)
        plt.xlabel('Distance between ball and center (m)')
        plt.ylabel('Number of Measurements')
        plt.title('Distribution of distance between ball and center')
    
    def K_E(self, num_frames):
        '''
        This function is used to generate the plot which is the kinetic energy
        of the system versus time.

        Parameters
        ----------
        num_frames : int
            The plot will show up after the number of frames you typed in.

        Returns
        -------
        None.

        '''
        time = []
        self.kinetic_energy = []
        for frame in range(num_frames):
            self.ke_frame = []
            for i in np.arange(0,len(self.balls)):
                ke = 0.5*self.balls[i].mass*np.linalg.norm(self.balls[i].velocity)**2
                self.ke_frame.append(ke)
            total_ke = sum(self.ke_frame)
            self.kinetic_energy.append(total_ke)
            t = self.next_collision()[0]
            self.t = self.t + t
            time.append(self.t)
        plt.plot(time,self.kinetic_energy)
        plt.xlabel('Time (s)')
        plt.ylabel('Kinetic Energy (J)')
        plt.title('System Kinetic Energy vs Time')

    def Momentum(self, num_frames):
        '''
        This function is used to generate the plot which is the momentum
        of the system versus time.

        Parameters
        ----------
        num_frames : int
            The plot will show up after the number of frames you typed in.

        Returns
        -------
        None.

        '''
        self.momentum_h = []
        self.momentum_v = []
        self.momentum = []
        time = []
        for frame in range(num_frames):
            self.momentum_h_frame = []
            self.momentum_v_frame = []
            self.momentum_frame = []
            for i in np.arange(0,len(self.balls)):
                momentum_h = self.balls[i].mass*self.balls[i].velocity[0]
                momentum_v = self.balls[i].mass*self.balls[i].velocity[1]
                speed = np.linalg.norm(self.balls[i].velocity)
                momentum = self.balls[i].mass*speed
                # momentum = np.sqrt(momentum_h**2 + momentum_v**2)
                self.momentum_h_frame.append(momentum_h)
                self.momentum_v_frame.append(momentum_v)
                self.momentum_frame.append(momentum)
            total_momentum_h = sum(self.momentum_h_frame)
            total_momentum_v = sum(self.momentum_v_frame)
            total_momentum = sum(self.momentum_frame)
            self.momentum_h.append(total_momentum_h)
            self.momentum_v.append(total_momentum_v)
            self.momentum.append(total_momentum)
            t = self.next_collision()[0]
            self.t = self.t + t
            time.append(self.t)
        plt.plot(time,self.momentum_h)
        plt.plot(time,self.momentum_v)
        plt.plot(time,self.momentum)
        plt.legend(['Horizontal momentum','Vertical momentum','Total momentum'],loc=0)
        plt.xlabel('Time (s)')
        plt.ylabel('Momentum (kgm/s)')
        plt.title('Momentum vs Time')
        

    def Pressure(self, num_frames):
        '''
        You do not need to consider this function. This function is only used
        to check if the pressure is a constant after lots of collisions.
        If you run this one, the plot which is pressure versus time will show
        up.

        Parameters
        ----------
        num_frames : int
            The plot will show up after the number of frames you typed in.

        Returns
        -------
        None.

        '''
        pressure_list = []
        time_list = []
        for frame in range(num_frames):
            t = self.min_collision_time()[0]
            self.t = self.t + t
            pressure = self.change_in_momentum / (self.t*np.pi*self.balls[0].radius**2)
            pressure_list.append(pressure)
            time_list.append(self.t)
            self.next_collision()
        plt.plot(time_list,pressure_list)
        plt.show()
        plt.title('pressure vs time')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (Pa)')
    
    def Pressure_Tem(self, num_frames):
        '''
        This function will generate the plot which is pressure versus
        temperature. Five different sets of temperatures and pressures will be
        calculated and used to find the best fit line. The default of the
        radius of a ball is 1 in this function.

        Parameters
        ----------
        num_frames : int
            After the number of frames you typed in, 
            the pressure will be calculated.
            The more frames, the more accurate the pressure. However, the more
            frames, the longer time it will take.

        Returns
        -------
        fit : list
            This returns us the gradient and y-intercept of the best fit line.

        '''
        pressure_list = []
        tem = []
        pre = []
        boltz_cons = 1.38 * 10 ** -23
        for frame in range(5):
            self.t = 0
            self.change_in_momentum = 0
            time_list = []
            print(frame)
            A=balls_generator(30, 1, 1,frame*2)
            self.balls = A
            speed_squared_list = []
            for i in range(len(self.balls)):
                speed_squared = np.linalg.norm(self.balls[i].velocity)**2
                speed_squared_list.append(speed_squared)
            speed_squared_sum = sum(speed_squared_list)
            ke = 0.5*1*speed_squared_sum/len(self.balls)
            tem.append(ke)
            for frame in range(num_frames):
                t = self.min_collision_time()[0]
                self.t = self.t + t
                print('time*A',self.t)
                print('changeinmo',self.change_in_momentum)
                pressure = self.change_in_momentum / (self.t*np.pi*self.balls[0].radius**2)
                pressure_list.append(pressure)
                time_list.append(self.t)
                self.next_collision()
            order = 9*len(pressure_list)/10
            pressure_new_list = pressure_list[int(order):]
            pressure_avg = sum(pressure_new_list)/(len(pressure_new_list))
            print(pressure_avg)
            pre.append(pressure_avg)
        fit,cov = np.polyfit(tem,pre,1,cov=True)
        p_pre = np.poly1d(fit)
        plt.plot(tem,pre,'o')
        plt.plot(tem,p_pre(tem),label='radius = 1')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Pressure (Pa)')
        plt.title('Pressure vs Temperature')
        plt.legend()
        plt.show()
        return fit
        
    def P_T_R(self, num_frames):
        '''
        This function will generate four different lines of pressure versus
        temperature. The radius of balls are different. The default radius for
        four lines are 0.1, 0.5, 0.8 and 1.

        Parameters
        ----------
        num_frames : int
            After the number of frames you typed in, 
            the pressure will be calculated.
            The more frames, the more accurate the pressure. However, the more
            frames, the longer time it will take.

        Returns
        -------
        None.

        '''
        fit_list = []
        for a in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1):
            pressure_list = []
            tem = []
            pre = []
            boltz_cons = 1.38 * 10 ** -23
            for frame in range(5):
                self.t = 0
                self.change_in_momentum = 0
                time_list = []
                print(frame)
                A=balls_generator(30, 1, a ,frame*2)
                self.balls = A
                speed_squared_list = []
                for i in range(len(self.balls)):
                    speed_squared = np.linalg.norm(self.balls[i].velocity)**2
                    speed_squared_list.append(speed_squared)
                speed_squared_sum = sum(speed_squared_list)
                ke = 0.5*1*speed_squared_sum/len(self.balls)
                tem.append(ke)
                for frame in range(num_frames):
                    t = self.min_collision_time()[0]
                    self.t = self.t + t
                    print('time*A',self.t)
                    print('changeinmo',self.change_in_momentum)
                    pressure = self.change_in_momentum / (self.t*np.pi*self.balls[0].radius**2)
                    pressure_list.append(pressure)
                    time_list.append(self.t)
                    self.next_collision()
                order = 9*len(pressure_list)/10
                pressure_new_list = pressure_list[int(order):]
                pressure_avg = sum(pressure_new_list)/(len(pressure_new_list))
                print(pressure_avg)
                pre.append(pressure_avg)
            fit,cov = np.polyfit(tem,pre,1,cov=True)
            fit_list.append(fit[0])
            # p_pre = np.poly1d(fit)
            # plt.plot(tem,pre,'o')
            # plt.plot(tem,p_pre(tem),label= a )
            # plt.xlabel('Temperature (K)')
            # plt.ylabel('Pressure (Pa)')
            # plt.title('Pressure vs Temperature')
            # plt.legend()
            # plt.show()
        radius = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        plt.plot(radius,fit_list,'x')
        plt.xlabel('Radius')
        plt.ylabel('Gradient of the PT line')
        plt.show()
            
    def Ideal_gas(self):
        '''
        You do not need to type in anything in this function.
        This function will automatically generate the plot which is pressure 
        versus temperature of ideal gas. If you run this afer running 'P_T_R'
        function, the line of ideal gas will be generated.

        Returns
        -------
        None.

        '''
        tem = np.array([0,20,40,60,80])
        pre = len(self.balls)*tem/(np.pi*self.balls[0].radius**2)
        plt.plot(tem,pre,label='Ideal')
        plt.legend()
        plt.show()


    def vel_his(self,num_frames,num_balls,sigma):
        '''
        This function generates the velocity histogram and comparision to the
        theoretical distribution.

        Parameters
        ----------
        num_frames : int
            After the number of frames you typed in, the histogram will show
            up.
            The more frame you type in, the more accurate the histogram.
        num_balls : int
            This is how many balls will be in the container.
        sigma : float
            The standard deviation of the velocity distribution.

        Returns
        -------
        None.

        '''
        vel = []
        self.balls = balls_generator(num_balls,1,1,sigma)
        for frames in range(num_frames):
            for i in np.arange(1,len(self.balls)):
                vel.append(np.linalg.norm(self.balls[i].velocity))
            self.next_collision()
        speed_squared_list = []
        
        for i in range(len(self.balls)):
            speed_squared = np.linalg.norm(self.balls[i].velocity)**2
            speed_squared_list.append(speed_squared)
        speed_squared_sum = sum(speed_squared_list)
        tem_ = 0.5*1*speed_squared_sum/len(self.balls)
        probability = []
        for i in range(len(vel)):
            p = vel[i]*np.exp(-0.5*self.balls[1].mass*vel[i]**2/tem_)/(num_frames*num_balls)
            probability.append(p)
        probability = np.array(probability)
        vel = np.array(vel)
        y,x,d=plt.hist(vel,40,density=True)
        fit,cov=curve_fit(func_gau,x[:40],y)
        plt.plot(x[:40],func_gau(x[:40],fit[0],fit[1],fit[2],fit[3]))
        plt.plot(x[:40],y,'x')
        plt.xlabel('Speed')
        plt.ylabel('Number of Measurements')
        plt.title('Histogram of Speed')
        plt.show()
        
    def waal(self,num_frames):
        '''
        This function calculates the constant a and b in van der Waals' law,
        and a plot of pressure versus temperature will generate. The default
        radius of a ball is 1.

        Parameters
        ----------
        num_frames : int
            After the number of frames you typed in, the constants and plot
            will show up.

        Returns
        -------
        None.

        '''
        fit = self.Pressure_Tem(num_frames)
        area = np.pi*self.balls[0].radius**2
        m = fit[0]
        y = fit[1]
        b = area/(len(self.balls)-1)-1/m
        a = -1*y*area**2/(len(self.balls)-1)**2
        print('a = ',a)
        print('b = ',b)



A=balls_generator(10, 1, 1, 8)
Test = Simulation(A)





    
                
        
        
        
        
        
        
