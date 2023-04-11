import numpy as np

# Define the Rosenbrock function
def log_prob(x, *args):
    a = args[0] 
    b = args[1] 
    x0,x1 = x
    if not (-2 <= x0 <= 2) or not (-1 <= x1 <= 3):
        return -np.inf
    return -((a - x0)**2 + b * (x1 - x0**2)**2)