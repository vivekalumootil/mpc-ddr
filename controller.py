import numpy as np
import math
from scipy.optimize import minimize
from parameters import *

class Robot:
    def __init__(self, timestep, horizon, world):
        self.timestep = timestep
        self.horizon = horizon
        self.world = world
        
    # u is 2x1, [lin_v, ang_v]
    # z is 3x1, [x, y, theta]
    def forward(self, z, u):
        forward_matrix = np.array([[math.cos(z[2]), 0], [math.sin(z[2]), 0], [0, 1]])
        f_z = z + np.dot(forward_matrix, u)*self.timestep 
        return f_z

    def move(self, u):
        self.world = self.forward(self.world, u)
    
    # m_z is 3x1, [a, b, c of ax+by+c=0]
    # z is 3x1, [x, y, theta]
    def error(self, u_seq, z, m_z):
        u_seq = u_seq.reshape(self.horizon, 2)
        cost = 0
        for i in range(self.horizon):
            u = u_seq[i]
            z = self.forward(z, u)
            cost += ((z[0]*m_z[0]+z[1]*m_z[1]+m_z[2]) ** 2)/(m_z[0]*m_z[0]+m_z[1]*m_z[1]) + u[1]*u[1]
        return cost 

    def optimize(self, z, m_z):
        bnd = [(MIN_LIN_VEL, MAX_LIN_VEL), (MIN_ANG_VEL, MAX_ANG_VEL)] * self.horizon
        result = minimize(self.error, args=(z, m_z), x0 = np.zeros((2*self.horizon)), method='SLSQP', bounds = bnd)     
        return result
