'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-03-26 20:10:43
 # @ Description: A MPC control simulation for the inverted pendulum. 
 '''

from time import time
import matplotlib.pyplot as plt  
import numpy as np
from numpy.lib import math

import cvxpy
from cvxpy.settings import OPTIMAL  


L_bar = 2  # length of bar 
M = 1.0 # [kg]
m = 0.3  
g = 9.8 

Q = np.diag([0.0, 1.0, 1.0, 0])
R = np.diag([0.01])
nx = 4  # number of states 
nu = 1 # number of input 
T = 30  # number of horizion length 
Delta_t = 0.1  # delta time

ANIMATION = True 

def get_model_matrix():

    # model Parameter 
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (L_bar * M), 0.0]
    ])
    A = np.eye(nx) + Delta_t * A  
    
    B = np.array([
        [0],
        [1 / M],
        [0],
        [1 / (L_bar * M)]
    ])     
    B = Delta_t * B 

    return A, B  

def flatten(x):
    """ get build-in list from matrix""" 
    return np.array(x).flatten()

def mpc_control(x0):
    x = cvxpy.Variable((nx, T+1))
    u = cvxpy.Variable((nu,T))

    A, B = get_model_matrix() 
    cost = 0.0 
    constraint = [] 
    for t in range(T): 
        cost += cvxpy.quad_form(x[:,t+1],Q) 
        cost += cvxpy.quad_form(u[:,t], R)
        constraint += [x[:,t+1] == A @ x[:,t] + B @ u[:,t]]  

    constraint += [x[:,0] == x0[:,0]]
    prob = cvxpy.Problem(cvxpy.Minimize(cost),constraint) 
    start = time() 
    prob.solve(verbose= False) 
    endtime = time()
    print("calculating time: {0} [sec]".format(endtime-start))

    if prob.status == cvxpy.OPTIMAL:
        ox = flatten(x.value[0, :])
        dx = flatten(x.value[1, :])
        theta = flatten(x.value[2, :])
        dtheta = flatten(x.value[3, :])

        ou = flatten(u.value[0, :])

    return ox, dx, theta, dtheta, ou

def simulation(x,u):
    A, B = get_model_matrix() 
    x = np.dot(A,x)  + np.dot(B, u) 
    return x 

def show_cart(xt, theta): 
    cart_w = 1 
    cart_h = 0.5 
    radius = 0.1 

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, L_bar * math.sin(-theta)])
    bx += xt
    by = np.matrix([cart_h, L_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")




def main():
    x0 = np.array([
        [0],
        [0],
        [0.3],
        [0]
    ])

    x = np.copy(x0)

    for i in range(50):
        ox, dx, otheta, dtheta, ou = mpc_control(x) 
        u = ou[0] 
        x = simulation(x,u)
        if ANIMATION: 
            plt.clf() 
            px = float(x[0]) 
            theta = float(x[2])
            show_cart(px, theta) 
            plt.xlim([-5, 2]) 
            plt.pause(0.001) 




if __name__ == '__main__':
    main() 
