
import time
import json
import numpy as np
from siriuspy.devices import CAXCtrl, DVF


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






## analysis functions ##



def fwhm_quick(data):
    threshold = 0.5*np.max(data)
    mask = data > threshold
    return np.sum(mask) * SCALE

def peak(data):
    return np.max(data)

def position(data):
    return np.argmax(data) * SCALE




def full_width(data, coords=None, hfactor=0.5):
    data = np.asarray(data)
    if coords is None:
        coords = np.arange(data.size)

    data_half = data-hfactor*np.max(data)

    sign = np.sign(data_half)
    idxleft, = np.nonzero(sign[:-1] + sign[1:] == 0)  # left, before zeros
    idxright = idxleft + 1                           # right, after zeros

    xl = coords[idxleft]  # x left
    xr = coords[idxright]  # x right
    yl = data_half[idxleft]    # y left
    yr = data_half[idxright]   # y right

    # linear interpolation to find the x-values at height
    zeros = (yr*xl-yl*xr)/(yr-yl)
    width = zeros[-1]-zeros[0]

    return width, zeros



def caustic_func(z, z0, amp, s0):
    """
    z: propagation position [m]
    z0: waist position [m]
    amp: amplitude [m]
    s0: spread factor [m]
    """
    return amp * np.sqrt( 1 + ((z-z0)/s0)**2 )

def caustic_processing(params, positions):
    z = np.linspace(positions[0], positions[-1], 2000)
    widthscfit = caustic_func(z, *params)

    z0 = params[0]
    dof = np.ptp(z[widthscfit <= 1.1*np.min(widthscfit)])/2
    zr = np.ptp(z[widthscfit <= np.sqrt(2)*np.min(widthscfit)])/2

    return z0, dof, zr



## beamline functions ##


CAX = CAXCtrl()


MAXERRORCOUNT = 5

def get_image(dvf: DVF):

    count = 0
    while count < MAXERRORCOUNT:
        try:
            if not dvf.acquisition_status:
                dvf.cmd_acquire_on()
            return dvf.image
        except Exception as err:
            print(f" WARNING. When trying to fetch image from DVF1: {err} ")
            time.sleep(2)
            count += 1
            if count < MAXERRORCOUNT:
                print("\n Repeating the procedure...\n")
            else:
                raise Exception("Client exception")



def current_config():
    return {
        'slit1': {
            'top': CAX.slit_A1.top_pos,
            'bottom': CAX.slit_A1.bottom_pos,
            'left': CAX.slit_A1.left_pos,
            'right': CAX.slit_A1.right_pos
        },
        'dvf1': {
            'acq_time': CAX.dvf_A1.acquisition_time,
            'expo_time': CAX.dvf_A1.exposure_time
        },
        'slit2': {
            'top': CAX.slit_B1.top_pos,
            'bottom': CAX.slit_B1.bottom_pos,
            'left': CAX.slit_B1.left_pos,
            'right': CAX.slit_B1.right_pos
        },
        'dvf2': {
            'acq_time': CAX.dvf_A1.acquisition_time,
            'expo_time': CAX.dvf_B1.exposure_time,
            'z_pos': CAX.dvf_B1.z_pos
        },
        'mirror': {
            'ry': CAX.mirror.ry_pos,
            'tx': CAX.mirror.tx_pos,
            'y1': CAX.mirror.y1_pos,
            'y2': CAX.mirror.y2_pos,
            'y3': CAX.mirror.y3_pos,
            'photocollector': CAX.mirror.photocurrent_signal
        }
    }

def save_beamline_config(filename, filedir):

    file = '/'.join([filedir,filename])
    if file.split('.')[-1] != 'json':
        file += '.json'

    config = current_config()

    with open(file, 'w') as f:
        json.dump(config, f, indent=4)