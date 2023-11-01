import numpy as np 
import math 
import time
def trig(a,b, vel_mag):
    angle = a/b
    sign = np.sign(b)
    sqrt = math.sqrt(1+angle**2)
    self_vel0 = (-vel_mag/sqrt)*sign
    self_vel1 = self_vel0*angle
    return (self_vel0,self_vel1)

def compute_values():
    data = np.array([[(0.0,0.0) for _ in range(202)] for k in range(202)])
    for i in range(202):
        for j in range(202):
            try:
                data[i][j] = trig(i-101, j-101, 1)
            except:
                data[i][j] = -1
    return data 
