import numpy as np
import matplotlib.pyplot as plt

def wrapAngle(radian):
    radian = radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))
    return radian