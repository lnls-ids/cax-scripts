


import numpy as np


## parameters ##

# dvf
SCALE = 1     # [um/px]
MAXERRORCOUNT = 5

# ry
STEP = 0.0001 # [mm]
DELAY = 2     # [s]

# slit1 limits
TOPMAX = 19.8
BOTTOMAX = 35.8
LEFTMAX = 44.56
RIGHTMAX = 45.9
#
TOPMIN = 15
BOTTOMIN = 31
LEFTMIN = 43.55
RIGHTMIN = 44.88
#
TOPMID = (TOPMIN + TOPMAX)/2
BOTTOMID = (BOTTOMIN + BOTTOMAX)/2
LEFTMID = (LEFTMIN + LEFTMAX)/2
RIGHTMID = (RIGHTMIN + RIGHTMAX)/2






## functions ##

# slit1



# analysis

def fwhm(data):
    threshold = 0.5*np.max(data)
    mask = data > threshold
    return np.sum(mask) * SCALE

def peak(data):
    return np.max(data)

def position(data):
    return np.argmax(data) * SCALE
