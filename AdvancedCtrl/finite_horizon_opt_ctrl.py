''' Finite Horizon Optimal Control '''

import numpy as np 
import matplotlib.pyplot as plt 

def CalcFiniteHorizonOptimalInput(A,B,Q,R,P,N,x0):
    ''' Calculate Finite Horizon OPtimal Input
    min: x'Px + sum(x'Qx + u'Pu) 
    s.t x(k+1) = Axk + Buk
    
    out: uopt '''

    # data check
    if A.shape[1] is not x0.shape[0]:
        print('Data Error')
        return None 
    elif B.shape[1] is not R.shape[1]:
        print('Data Error')
        return None 

    sx = np.eye(A.ndim) 
    su = np.zeros((A.ndim,B.shape[1] * N))

    for i in range(N):
        # generate sx 
        An = np.linalg.matrix_power(A,i+1) 
        sx = np.r_[sx, An]

        # generate su
        tmp = None 
        for ii in range(i+1):
            tm = np.linalg.matrix_power(A,ii) *B
            if tmp is None: 
                tmp = tm 

            else:
                tmp = np.c_[tm,tmp] 

        for ii in np.arange(i,N-1):
            tm = np.zeros(B.shape)
            if tmp is None:
                tmp = tm 
            else:
                tmp = np.c_[tmp,tm]
        
        su = np.r_[su,tmp] 

    tm1 = np.eye(N+1) 
    tm1[N,N] = 0 
    tm2 = np.zeros((N+1, N+1))
    tm2[N,N] = 1 
    Qbar = np.kron(tm1, Q) + np.kron(tm2,P)
    Rbar = np.kron(np.eye(N),R) 

    uopt = -(su.T *Qbar *su + Rbar).I * su.T * Qbar * sx * x0 
    costBa = x0.T *(sx.T * Qbar * sx - sx.T *Qbar * su * (su.T*Qbar*su + Rbar).I *su.T * Qbar*sx) 

    return uopt

    

if __name__ == '__main__':

    A  = np.matrix([[0.77, -0.35],[0.49, 0.91]])
    B = np.matrix([0.04, 0.15]).T 
    x0 = np.matrix([1,-1]).T 
    Q = np.matrix([[500,0],[0,100]])
    R = np.matrix([1])
    P = np.matrix([[1500,0],[0,100]])
    N = 20 

    uopt = CalcFiniteHorizonOptimalInput(A,B,Q,R,P,N,x0) 

    #simulation
    u_history=[]
    x1_history=[]
    x2_history=[]
    x=x0
    for u in uopt:
        u_history.append(float(u[0]))
        x=A*x+B*u
        x1_history.append(float(x[0]))
        x2_history.append(float(x[1]))

    plt.plot(u_history,"-r",label="input")
    plt.plot(x1_history,"-g",label="x1")
    plt.plot(x2_history,"-b",label="x2")
    plt.grid(True)
    plt.legend()
    plt.show()