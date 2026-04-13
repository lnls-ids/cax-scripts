"""Utility functions for CAX mirror scanning scripts."""

import time
import json
import numpy as np
from siriuspy.devices import CAXCtrl, DVF
import matplotlib.pyplot as plt
import epics

# # parameters ##

# dvf
SCALE = 1     # [um/px]
MAXERRORCOUNT = 3

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

def snapshot_storage_ring():
    """Get current storage ring status as a dictionary."""
    return {
        'current': epics.caget('SI-13C4:DI-DCCT:Current-Mon')
    }

def snapshot_dvf(dvf, *, include_imgproc=False):
    """Capture a DVF snapshot: image, camera settings, and beam diagnostics.

    Args:
        dvf: a DVF (or DVFImgProc / CAXDtc) device instance.
        include_imgproc: if True, read beam-fit PVs (centroid, widths,
            tilt angle, ellipse sigmas).  Only meaningful for DVFImgProc
            subclasses.

    Returns:
        dict with 'image', 'exposure_time', 'acquisition_time', and
        optionally 'beam' sub-dict.
    """
    snap = {
            'acq_time'      : dvf.acquisition_time,
            'expo_time'     : dvf.exposure_time,
            'image'         : get_image(dvf),
    }

    if include_imgproc:
        snap.update({
            # --- centroid (1-D Gaussian fit means, in pixels) ---
            'roix_fit_mean'  : dvf.roix_fit_mean,
            'roiy_fit_mean'  : dvf.roiy_fit_mean,

            # --- beam widths (1-D Gaussian fit sigmas, in pixels) ---
            'roix_fit_sigma' : dvf.roix_fit_sigma,
            'roiy_fit_sigma' : dvf.roiy_fit_sigma,

            # --- FWHM of the ROI projections (pixels) ---
            'roix_fwhm'      : dvf.roix_fwhm,
            'roiy_fwhm'      : dvf.roiy_fwhm,

            # --- 2-D ellipse fit (from second-moment SVD) ---
            'fit_angle'      : dvf.fit_angle,     # tilt angle [deg]
            'fit_sigma1'     : dvf.fit_sigma1,    # major-axis sigma [px]
            'fit_sigma2'     : dvf.fit_sigma2,    # minor-axis sigma [px]

            # --- fit quality ---
            'roix_fit_error' : dvf.roix_fit_error,
            'roiy_fit_error' : dvf.roiy_fit_error,
        })
    return snap


def snapshot_machine_state(cax: CAXCtrl):
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
            'cs_rx'         : [caxm.cs_rx_mon,
                               caxm.cs_rx_lolm,
                               caxm.cs_rx_hilm,
                               caxm.cs_rx_enbl],
            'cs_rz'         : [caxm.cs_rz_mon,
                               caxm.cs_rz_lolm,
                               caxm.cs_rz_hilm,
                               caxm.cs_rz_enbl],
            'cs_tx'         : [caxm.cs_tx_mon,
                               caxm.cs_tx_lolm,
                               caxm.cs_tx_hilm,
                               caxm.cs_tx_enbl],
            'cs_ty'         : [caxm.cs_ty_mon,
                               caxm.cs_ty_lolm,
                               caxm.cs_ty_hilm,
                               caxm.cs_ty_enbl],
            'photocollector': caxm.photocurrent_signal
        }

    caxsA1 = cax.slit_A1
    slitA1_status = {
            'top'           : [caxsA1.top_mon,
                               caxsA1.top_lolm,
                               caxsA1.top_hilm,
                               caxsA1.top_enbl],
            'bottom'        : [caxsA1.bottom_mon,
                               caxsA1.bottom_lolm,
                               caxsA1.bottom_hilm,
                               caxsA1.bottom_enbl],
            'left'          : [caxsA1.left_mon,
                               caxsA1.left_lolm,
                               caxsA1.left_hilm,
                               caxsA1.left_enbl],
            'right'         : [caxsA1.right_mon,
                               caxsA1.right_lolm,
                               caxsA1.right_hilm,
                               caxsA1.right_enbl]
        }

    caxsB1 = cax.slit_B1
    slitB1_status = {
            'top'           : [caxsB1.top_mon,
                               caxsB1.top_lolm,
                               caxsB1.top_hilm,
                               caxsB1.top_enbl],
            'bottom'        : [caxsB1.bottom_mon,
                               caxsB1.bottom_lolm,
                               caxsB1.bottom_hilm,
                               caxsB1.bottom_enbl],
            'left'          : [caxsB1.left_mon,
                               caxsB1.left_lolm,
                               caxsB1.left_hilm,
                               caxsB1.left_enbl],
            'right'         : [caxsB1.right_mon,
                               caxsB1.right_lolm,
                               caxsB1.right_hilm,
                               caxsB1.right_enbl]
        }

    # DVF1 status accounts for caustic scans.
    dvfA1_status = snapshot_dvf(cax.dvf_A1, include_imgproc=False)

    # caustic_status = {
    #     'z_pos' : [dvf2.z_pos,
    #                dvf2.z_min,
    #                dvf2.z_max,
    #                dvf2["PP01:E.CNEN"]]  # Bypass to get enable status of z_pos
    #     }
    # caustic_status.update(snapshot_dvf(dvf2, include_imgproc=False))

    # DVF2 status accounts for lens and caustic scans.
    dvfB1  = cax.dvf_B1
    dvf_B1_status = {
        'z_pos'    : [dvfB1.z_mon,
                      dvfB1.z_lolm,
                      dvfB1.z_hilm,
                      dvfB1.z_enbl],  # Bypass to get enable status of z_pos
        'lens_pos' : [dvfB1.lens_mon,
                      dvfB1.lens_lolm,
                      dvfB1.lens_hilm,
                      dvfB1.lens_enbl],  # Bypass to get enable status of lens_pos
        }
    dvf_B1_status.update(snapshot_dvf(dvfB1, include_imgproc=True))

    sirius_status = snapshot_storage_ring()

    return {
        'mirror'  : mirror_status,
        'slit_A1' : slitA1_status,
        'slit_B1' : slitB1_status,
        'dvf_A1'  : dvfA1_status,
        'dvf_B1'  : dvf_B1_status,
        'sirius'  : sirius_status,
    }


def save_beamline_config(filename, filedir):
    """Save current beamline configuration to a JSON file."""
    file = '/'.join([filedir, filename])
    if file.split('.')[-1] != 'json':
        file += '.json'

    config = snapshot_machine_state()

    with open(file, 'w') as f:
        json.dump(config, f, indent=4)


def data_save(h5file, scaname, setmetadata, dvfimg, devname='dvf'):
    """Save scan data (image + metadata) to an HDF5 group.

    .. deprecated:: Use :func:`save_step` instead for step dicts produced
       by :func:`snapshot_machine_state`.

    Args:
        h5file: HDF5File instance (or None to skip saving).
        scaname: name of the HDF5 group for this scan step.
        setmetadata: metadata dict attached to the dataset.
        dvfimg: image array to save.
        devname: dataset name inside the group.
    """
    if h5file:
        h5file.save_group(grpname=scaname)
        h5file.save_dataset(
            grpname=scaname, dsetdata=dvfimg,
            dsetname=devname, dsetmetadata=setmetadata,
        )


# --- DVF sub-keys that hold images (saved as datasets, not attributes) ---
_DVF_IMAGE_KEY = 'image'

# Keys inside a step dict whose values are DVF snapshot dicts (contain images).
_DVF_KEYS = ('dvf_A1', 'dvf_B1')


def _flatten_dict(d, prefix=''):
    """Flatten a (possibly nested) dict using '.' as separator.

    Entries whose values are numpy arrays are skipped (images are saved
    separately as HDF5 datasets).
    """
    flat = {}
    for key, val in d.items():
        full_key = f'{prefix}{key}' if not prefix else f'{prefix}.{key}'
        if isinstance(val, dict):
            flat.update(_flatten_dict(val, prefix=full_key))
        elif val is None:
            flat[full_key] = 'N/A'
        elif not isinstance(val, np.ndarray):
            flat[full_key] = val
    return flat


def save_step(h5file, step, step_index=None):
    """Persist one scan-step dict (from snapshot_machine_state) to HDF5.

    HDF5 layout produced for each step::

        /scan-NNNN                          ← group  (one per step)
            @step          = 0              ← group attrs: all scalars
            @scan_type     = 'mirror'       ←   and flat sub-dicts are
            @mirror.tx     = 1.234          ←   flattened with '.' separator
            @slit_A1.top   = 19.5
            ...
            dvf1           (dataset)        ← gzip-compressed image array
                @exposure_time    = 0.01    ← dataset attrs: DVF scalars
                @acquisition_time = 0.05
            dvf2           (dataset)        ← gzip-compressed image array
                @exposure_time    = 0.01
                @beam.fit_angle   = 12.3    ← beam sub-dict flattened
                ...

    Args:
        h5file: HDF5File instance (or None to skip saving).
        step: dict as returned by ``device_scan`` (step index +
            :func:`snapshot_machine_state` keys).
        step_index: explicit step number for the group name.  If *None*,
            ``step.get('step', 0)`` is used.
    """
    if not h5file:
        return

    idx = step_index if step_index is not None else step.get('step', 0)
    grpname = f'scan-{idx:04d}'

    # --- Separate DVF dicts (contain images) from everything else ---
    non_dvf = {k: v for k, v in step.items() if k not in _DVF_KEYS}
    grp_attrs = _flatten_dict(non_dvf)

    h5file.save_group(grpname=grpname, grpmetadata=grp_attrs)

    # --- Save each DVF as a dataset inside the group ---
    for dvf_key in _DVF_KEYS:
        dvf_dict = step.get(dvf_key)
        if dvf_dict is None:
            continue

        image = dvf_dict[_DVF_IMAGE_KEY]
        dset_attrs = _flatten_dict(
            {k: v for k, v in dvf_dict.items() if k != _DVF_IMAGE_KEY}
        )
        h5file.save_dataset(
            grpname=grpname,
            dsetname=dvf_key,
            dsetdata=image,
            dsetmetadata=dset_attrs,
        )


# Slit limits defined from DVF images with slits fully open.
# This avoids long motor moves and discrepancies between
# hard limits and those defined in the front end.
SLIT_A1_TOP_LIMS = {
    'TOP'    : 19.5,
    'BOTTOM' : 37.5,
    'LEFT'   : 47.5,
    'RIGHT'  : 46.5,
}


def slit_open(device_mv, cax_status):
    """Open the slits."""
    devname = device_mv.devname
    slit_status = [
        cax_status[devname][slit][0]
        for slit in ('top', 'bottom', 'left', 'right')
        ]

    device        = device_mv.device
    device.top    = SLIT_A1_TOP_LIMS['TOP']
    device.bottom = SLIT_A1_TOP_LIMS['BOTTOM']
    device.left   = SLIT_A1_TOP_LIMS['LEFT']
    device.right  = SLIT_A1_TOP_LIMS['RIGHT']

    return slit_status


IMG_THRESHOLD = 100


def image_show_slit_boundary(img):
    """Determine slit boundaries from the image and display them."""
    roi = img > IMG_THRESHOLD
    hor, vert = img.shape

    left = min(vert - 1, np.argmax(roi.sum(axis=0) > 0))
    right = max(0, vert - 1 - np.argmax(roi[:, ::-1].sum(axis=0) > 0))

    top = min(hor - 1, np.argmax(roi.sum(axis=1) > 0))
    bottom = max(0, hor - 1 - np.argmax(roi[::-1, :].sum(axis=1) > 0))

    fig, ax = plt.subplots()
    pix_to_um = 0.48
    print("\n ** Current slit boundaries at DVF (in microns):\n"
          f"      top: {top    * pix_to_um:.2f},\n"
          f"   bottom: {bottom * pix_to_um:.2f},\n"
          f"     left: {left   * pix_to_um:.2f},\n"
          f"    right: {right  * pix_to_um:.2f}\n")
    import matplotlib.patches as patches
    rect = patches.Rectangle((left, top), right - left, bottom - top,
                            linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.imshow(img)
    plt.show(block=False)
    plt.pause(0.1)

    return [top, bottom, left, right]
