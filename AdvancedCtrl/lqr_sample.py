''' 
Linear Quadratic Regulator sample code 
 '''

import time 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.linalg as la 

Sim_time = 3.0 
Dt = 0.1 

# x[k+1] = Ax[k] + Bu[k]

A = np.matrix([[1, 1.0], [0,1]])
B = np.matrix([0.0, 1]).T 
Q = np.matrix([[1.0, 0.0],[0.0,0.0]])
R = np.matrix([[1]])
Kopt = None 

def process(x,u):
    return A * x + B * u 

def solve_DARE_with_iteration(A,B,Q,R):
    ''' Solve discrete time Algebraic Riccati eq. '''
    X = Q  
    maxiter  = 150 
    eps = 0.01 
    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * la.inv(R +B.T * X *B) *B.T * X * A + Q 
        if (abs(Xn - X )).max() < eps:
            X = Xn 
            break 
        X = Xn 
    return Xn 



def dlqr_with_iteration(Ad, Bd, Q, R): 
    ''' Solve discrete time lqr controller
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    '''

    X = solve_DARE_with_iteration(Ad, Bd, Q, R)

    K = np.matrix(la.inv(Bd.T * X * Bd + R) * (Bd.T * X * Ad))

    return K 

def lqr_ref_tracking(x, xref, uref):
    global Kopt
    if Kopt is None: 
        # start = time.time() 
        # Kopt = dlqr_with_iteration(A,B,np.eye(2),np.eye(1))
        Kopt = dlqr_with_iteration(A,B,Q,R)

        # elapsed_time = time.time() - start
        # print('elapsed_time: {0}'.format(elapsed_time) + '[sec]')

    u = -uref - Kopt * (x - xref) 
    return u 

def main_reference_tracking():
    t = 0.0 
    x = np.matrix([3,1]).T 
    u = np.matrix([0])
    xref = np.matrix([1,0]).T 
    uref = 0.0 

    time_history = [0.0]
    x1_history = [x[0,0]]
    x2_history = [x[1,0]]
    u_history  = [0]

    while t <= Sim_time:
        u = lqr_ref_tracking(x, xref, uref)
        u0 = float(u[0,0])  # why [0,0] here 
        x = process(x, u0) 
        x1_history.append(x[0,0])
        x2_history.append(x[1,0])

        u_history.append(u0) 
        time_history.append(t)
        t += Dt

    plt.plot(time_history,u_history,'-r',label = 'input')
    plt.plot(time_history,x1_history,'-b',label = 'x1')
    plt.plot(time_history,x2_history,'-g', label = 'x2')
    xref0 = [xref[0,0] for i in range(len(time_history))]
    xref1 = [xref[1,0] for i in range(len(time_history))]
    plt.plot(time_history, xref0, '--b', label = 'target x1')
    plt.plot(time_history, xref1, '--g', label = 'target x2')
    plt.grid(True)
    plt.title('LQR')
    plt.legend()
    plt.show() 



if __name__ == '__main__':
    main_reference_tracking()
    print('Done')