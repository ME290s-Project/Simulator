import basic_pid 
import time 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.interpolate import make_interp_spline

def test_pid(P = 0.2, I= 0.0, D = 0.0, L = 100):
    """ test-PID 
        for i in range(1, END):
            pid.update(feedback) 
            output = pid.output 
            if pid.Setpoint > 0: 
                feedback += (output - (1/i))
            if i > 9:
                pid.Setpoint = 1 
            time.sleep(0.02 )
    L: end time 
    """

    pid = basic_pid.PID(P,I,D)
     
    pid.SetPoint = 0.0 
    pid.setSampleTime(0.01) 

    END = L 
    feedback = 0 
    feedback_list = [] 
    time_list = [] 
    setpoint_list = [] 
    for i in range(1, END):
        pid.update(feedback)
        output = pid.output
        if pid.SetPoint > 0:
            feedback += (output - 1/i) 
        if i > 9:
            pid.SetPoint = 1 
        time.sleep(0.02) 

        feedback_list.append(feedback) 
        setpoint_list.append(pid.SetPoint) 
        time_list.append(i) 

    time_sm = np.array(time_list) 
    time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)
    helper = make_interp_spline(time_list,feedback_list)
    feedback_smooth = helper(time_smooth) 

    plt.plot(time_smooth, feedback_smooth) 
    plt.plot(time_list,setpoint_list) 
    plt.xlim((0,L))
    # plt.ylim((min(feedback_list),max(feedback_list)))
    plt.xlabel('time(s)')
    plt.ylabel('PID(PV)')
    plt.title('TEST PID')
    
    plt.grid(True) 
    plt.show() 


if __name__ == '__main__':

    test_pid(1.2,0,0.002,50)