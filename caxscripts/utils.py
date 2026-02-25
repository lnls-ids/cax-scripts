"""Utility functions for CAX mirror scanning scripts."""

import time
import json
import numpy as np
from siriuspy.devices import CAXCtrl, DVF


# # parameters ##

# dvf
SCALE = 1     # [um/px]
MAXERRORCOUNT = 5

# ry
STEP = 0.0001  # [mm]
DELAY = 2      # [s]

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


# ## analysis functions ##


def fwhm_quick(data):
    """Calculate full width at half maximum quickly."""
    data = np.asarray(data)
    threshold = 0.5*np.max(data)
    mask = data > threshold
    return np.sum(mask) * SCALE


def peak(data):
    """Calculate peak value."""
    return np.max(data)


def position(data):
    """Calculate position of the peak."""
    return np.argmax(data) * SCALE


def full_width(data, coords=None, hfactor=0.5):
    """Calculate full width at given height factor."""
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
    """Calculate caustic function.

    z: propagation position [m]
    z0: waist position [m]
    amp: amplitude [m]
    s0: spread factor [m]
    """
    return amp * np.sqrt(1 + ((z-z0)/s0)**2)


def caustic_processing(params, positions):
    """Process caustic scan results."""
    z = np.linspace(positions[0], positions[-1], 2000)
    widthscfit = caustic_func(z, *params)

    z0 = params[0]
    dof = np.ptp(z[widthscfit <= 1.1*np.min(widthscfit)])/2
    zr = np.ptp(z[widthscfit <= np.sqrt(2)*np.min(widthscfit)])/2

    return z0, dof, zr


# # beamline functions ##


# CAX = CAXCtrl()


def get_image(dvf: DVF):
    """Get image from DVF with retries on failure."""
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
                raise Exception("Client exception") from err


def current_config(cax: CAXCtrl):
    """Get current beamline configuration as a dictionary."""
    caxm  = cax.mirror
    mirror_status = {
            'tx'            : [caxm.tx_mon,
                               caxm.tx_lolm,
                               caxm.tx_hilm,
                               caxm.tx_enbl],
            'ry'            : [caxm.ry_mon,
                               caxm.ry_lolm,
                               caxm.ry_hilm,
                               caxm.ry_enbl],
            'y1'            : [caxm.y1_mon,
                               caxm.y1_lolm,
                               caxm.y1_hilm,
                               caxm.y1_enbl],
            'y2'            : [caxm.y2_mon,
                               caxm.y2_lolm,
                               caxm.y2_hilm,
                               caxm.y2_enbl],
            'y3'            : [caxm.y3_mon,
                               caxm.y3_lolm,
                               caxm.y3_hilm,
                               caxm.y3_enbl],
            'rx'            : [caxm.cs_rx_mon,
                               caxm.cs_rx_lolm,
                               caxm.cs_rx_hilm,
                               caxm.cs_rx_enbl],
            'rz'            : [caxm.cs_rz_mon,
                               caxm.cs_rz_lolm,
                               caxm.cs_rz_hilm,
                               caxm.cs_rz_enbl],
            'ty'            : [caxm.cs_ty_mon,
                               caxm.cs_ty_lolm,
                               caxm.cs_ty_hilm,
                               caxm.cs_ty_enbl],
            'photocollector': caxm.photocurrent_signal
        }

    caxs1 = cax.slit_A1
    cpv1  = caxs1.PVS
    slit1_status = {
            'top'           : [caxs1.top_pos,
                               caxs1[cpv1.TOP_LOLM],
                               caxs1[cpv1.TOP_HILM],
                               caxs1[cpv1.TOP_ENBL]],
            'bottom'        : [caxs1.bottom_pos,
                               caxs1[cpv1.BOTTOM_LOLM],
                               caxs1[cpv1.BOTTOM_HILM],
                               caxs1[cpv1.BOTTOM_ENBL]],
            'left'          : [caxs1.left_pos,
                               caxs1[cpv1.LEFT_LOLM],
                               caxs1[cpv1.LEFT_HILM],
                               caxs1[cpv1.LEFT_ENBL]],
            'right'         : [caxs1.right_pos,
                               caxs1[cpv1.RIGHT_LOLM],
                               caxs1[cpv1.RIGHT_HILM],
                               caxs1[cpv1.RIGHT_ENBL]]
        }

    caxs2 = cax.slit_B1
    cpv2  = caxs2.PVS
    slit2_status = {
            'top'           : [caxs2.top_pos,
                               caxs2[cpv2.TOP_LOLM],
                               caxs2[cpv2.TOP_HILM],
                               caxs2[cpv2.TOP_ENBL]],
            'bottom'        : [caxs2.bottom_pos,
                               caxs2[cpv2.BOTTOM_LOLM],
                               caxs2[cpv2.BOTTOM_HILM],
                               caxs2[cpv2.BOTTOM_ENBL]],
            'left'          : [caxs2.left_pos,
                               caxs2[cpv2.LEFT_LOLM],
                               caxs2[cpv2.LEFT_HILM],
                               caxs2[cpv2.LEFT_ENBL]],
            'right'         : [caxs2.right_pos,
                               caxs2[cpv2.RIGHT_LOLM],
                               caxs2[cpv2.RIGHT_HILM],
                               caxs2[cpv2.RIGHT_ENBL]]
        }

    dvf1  = cax.dvf_A1
    dvf1_status = {
            'acq_time'      : dvf1.acquisition_time,
            'expo_time'     : dvf1.exposure_time
        }

    dvf2  = cax.dvf_B1
    dvf2_status = {
            'acq_time'      : dvf2.acquisition_time,
            'expo_time'     : dvf2.exposure_time,
            'z_pos'         : dvf2.z_pos
        }

    return {
        'mirror': mirror_status,
        'slit1': slit1_status,
        'slit2': slit2_status,
        'dvf1': dvf1_status,
        'dvf2': dvf2_status
    }


def save_beamline_config(filename, filedir):
    """Save current beamline configuration to a JSON file."""
    file = '/'.join([filedir, filename])
    if file.split('.')[-1] != 'json':
        file += '.json'

    config = current_config()

    with open(file, 'w') as f:
        json.dump(config, f, indent=4)
