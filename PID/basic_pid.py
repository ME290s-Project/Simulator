""" Simple PID controller """
import time 

class PID():
    """ PID controller """
    def __init__(self, P = 0.2, I = 0.0, D = 0.0, current_time = None):
        self.Kp = P 
        self.Ki = I 
        self.Kd = D 

        self.sample_time = 0.00 
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time 

        self.clear()

    def clear(self):
        """ Clears PID computations con coefficients """
        self.SetPoint = 0.0 
        self.PTerm = 0.0 
        self.ITerm = 0.0 
        self.DTerm = 0.0 
        self.last_error = 0.0 

        # windup Guard 
        self.int_error = 0.0 
        self.windup_guard = 20.0 
        self.output = 0.0 

    def update(self, feedback_value, current_time = None):
        """ calculate PID value for given reference feedback """
        error = self.SetPoint - feedback_value 
        self.current_time = current_time if current_time is not None else time.time() 
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error 

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error 
            self.ITerm += error * delta_time 

            if (self.ITerm < - self.windup_guard):
                self.ITerm = - self.windup_guard 
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard 

            self.DTerm = 0.0 
            if delta_time > 0:
                self.DTerm = delta_error / delta_time 

            self.last_time = self.current_time 
            self.last_error = error 
            self.output = self.PTerm + (self.Ki *self.ITerm) + self.Kd * self.DTerm 

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """ Integral windup, also known as intergrator windup or reset windup
        refers to the situation in a PID feedback controller where 
        a large change in setpoint occurs (say a positive gain) 
        and the integral terms accumulates a significant error 
        during the rise (windup), thus overshooting and continuing to increase
        as this accumulated error is unwound
        (offset by errors in the other direction)
        The specific problem is the excess overshooting
        """
        self.windup_guard = windup 
    
    def setSampleTime(self,sample_time):
        self.sample_time = sample_time 