"""Set of methods to analyze scan data and extract beam properties.

This module provides functions to analyze Caracara data obtained
from scanning with its motors. All data is expected to be stored
in HDF5 files. The functions here defined are designed to read the
data, extract beam properties such as centroids, intensities and
FWHMs, and visualize the results.
The main focus is on analyzing the beam profiles obtained from
scans and understanding how they change with respect to the
scanned variable (e.g., variations in Tx).
Main functions:

- beam_centroid: Return centroids of the beam profiles from the data dict.

- beam_fwhm: Return fwhm of the beam profiles from the data dict.

- beam_intensity: Return the total intensity of the beam profiles from
    the data dict.

- beam_properties: Return properties of the beam profiles from the data dict.

- caustic_analysis: Perform caustic analysis on the given dataset.

- centroid_plot: Plot the beam profile images and their centroids in a dataset.

- correlate: Calculate the normalized cross-correlation between
    two 1D arrays.

- data_from_h5_files: Read multiple HDF5 files into a nested dict.

- dataset_plot: Plot the behavior of an observable across scans for
    a single dataset.

- files_in_directory: List files in a directory matching a regex pattern.

- fwhm_plot: Plot the beam profile images and their FWHMs in a dataset.

- get_scan_data: Extract observable and variable values across scans.

- h5_to_dict: Read an HDF5 file into a nested dict.

- observable_data: Extract the behavior of a variable across scans
    in a dataset.

- observable_statistics: Calculate statistics of an observable across scans
    in a dataset.

- plot_double_observable: Plot two-component observables in separate subplots.

- scan_plot: Plot the behavior of an observable across scans for each dataset.
"""
import h5py
import numpy as np
import os
import re

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from IPython.display import HTML
from IPython.display import display as ipydisplay

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from . import utils

# Threshold for peak-to-average ratio acceptance of image.
THRESHOLD = 100

#
# Scan parameter extraction methods.
#


def _get_variable_metadata(data, dev_motor):
    """Helper function to extract the variable metadata from the dataset.

    Args:
        data (dict): Dict containing data for a single step of a scan.
        dev_motor (str): The device and motor being scanned
            (e.g., 'mirror.rx').

    Returns:
        list: Metadata of the scanned variable: [value, lolm, hilm, enable].
    """
    # Separate device and motor from the input string.
    device, motor = dev_motor.split('.')

    try:
        # First try to get data directly from the scan attributes.
        if data['attrs'].get(dev_motor) is not None:
            meta = data['attrs'].get(dev_motor)
        # In the case of DVFs, the scanned variable is stored in
        # the device group attributes.
        elif data.get(device, None) is not None:
            meta = data[device]['attrs'].get(motor)
    except (KeyError, TypeError, ValueError) as err:
        meta = None
        raise ValueError(f"Could not extract scanned variable metadata"
                         f"for {dev_motor}") from err
    return meta

#
# Beam properties extraction methods and analysis.
#


def beam_centroid(datascan, dev_motor, droi=4, threshold=THRESHOLD):
    """Return centroids of the beam profiles from the data dict.

    Args:
        datascan (dict): Nested dict containing the set of one full scan.
        dev_motor (str): The device and motor being scanned
            (e.g., 'mirror.rx').
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.
        threshold (float): The minimum peak-to-average ratio required to
            consider the beam visible. Default is 10.

    Returns:
        dict: A dict mapping scan numbers to (cx, cy) centroids.
    """
    # FWHM to sigma conversion factor.
    f2sig = 2 * np.sqrt(2 * np.log(2))

    # Calculate beam properties for all scans.
    _, beam_propties = beam_properties(datascan, dev_motor, droi, threshold)

    scans, xvals, centrs, sigmas  = [], [], [], []
    for sc, values in beam_propties.items():
        scans.append(sc)
        xvals.append(values[0])
        centrs.append(values[1])

        # Estimate sigma from FWHM for a Gaussian profile.
        fwhms = values[2]
        sigmas.append(fwhms / f2sig)

    return (np.array(scans), np.array(xvals),
            np.array(centrs), np.array(sigmas))


def beam_fwhm(datascan, dev_motor, droi=4, threshold=THRESHOLD):
    """Return fwhm of the beam profiles from the data dict.

    Args:
        datascan (dict): Nested dict containing the set of one full scan
            (one pass).
        dev_motor (str): The device and motor being scanned (e.g., 'sample.x').
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.
        threshold (float): The minimum peak-to-average ratio required to
            consider the beam visible. Default is 10.

    Returns:
        dict: A dict mapping scan numbers to (fx, fy) fwhm's.
    """
    fwhms = {}
    _, beam_propties = beam_properties(datascan, dev_motor, droi, threshold)

    for sc in beam_propties.keys():
        xval   = beam_propties[sc][0]
        fx, fy = beam_propties[sc][2]

        fwhms[sc] = [xval, [fx, fy]]

    return fwhms


def beam_intensity(datascan, dev_motor, droi=4, threshold=THRESHOLD):
    """Return the total intensity of the beam profiles from the data dict.

    Args:
        datascan (dict): Nested dict containing the set of one full scan
            (one pass).
        dev_motor (str): The device and motor being scanned (e.g., 'sample.x').
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.
        threshold (float): The minimum peak-to-average ratio required to
            consider the beam visible. Default is 10.

    Returns:
        dict: A dict mapping scan numbers to total intensity.
    """
    beam_images, beam_propties = beam_properties(datascan, dev_motor,
                                                 droi, threshold)
    exptime = datascan['scan-0000']['dvf_B1']['attrs']['expo_time']

    intensities = {}
    droi = 2
    for sc in beam_images.keys():
        img            = beam_images[sc]
        xval           = beam_propties[sc][0]
        cx, cy         = beam_propties[sc][1]
        fwhm_x, fwhm_y = beam_propties[sc][2]

        peak = np.mean(img[cy - droi : cy + droi + 1,
                           cx - droi : cx + droi + 1])

        mask = img > (peak / 2)
        area_mask  = np.sum(mask)
        img_masked = np.where(mask, img, 0)
        area_img_masked = np.sum(img_masked)

        intensity_by_mask = (area_img_masked / (area_mask * exptime)
                             if area_mask != 0 else 0)

        peak /= exptime
        peak_fwhm_norm = (peak / (fwhm_x * fwhm_y)
                          if fwhm_x * fwhm_y != 0 else 0)

        intensities[sc] = [xval, [peak, intensity_by_mask, peak_fwhm_norm]]

    return intensities


def beam_properties(datascan, dev_motor, droi=4, threshold=THRESHOLD):
    """Return properties of the beam profiles from the data dict.

    Args:
        datascan (dict): Nested dict containing the set of one full scan.
        dev_motor (str): The device and motor being scanned (e.g., 'sample.x').
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.
        threshold (float): The minimum peak-to-average ratio required to
            consider the beam visible. Default is 10.

    Returns:
        beam_propties : A dict mapping scan numbers to (cx, cy), (fx, fy)
            centroids and FWHMs.
        beam_images   : A dict mapping scan numbers to DVF images of the beam,
            only scans where the beam is visible are returned.
    """
    beam_propties = {}
    beam_images   = {}
    xval          = []

    for scan, data in datascan.items():
        # Extract scanning index and observable value.
        sc = int(scan.split('-')[-1])

        # Extract the scanned variable value for this scan step.
        xval = float(_get_variable_metadata(data, dev_motor)[0])

        # Get image data and calculate centroid.
        img = data['dvf_B1']['data']

        # Centroids.
        # cx = np.sum(img, axis=0).argmax()
        # cy = np.sum(img, axis=1).argmax()
        cx_sum = np.sum(img, axis=0)
        cy_sum = np.sum(img, axis=1)

        xsmooth = savgol_filter(cx_sum, window_length=21, polyorder=2)
        ysmooth = savgol_filter(cy_sum, window_length=21, polyorder=2)

        cx = np.argmax(xsmooth)
        cy = np.argmax(ysmooth)

        # FWHMs
        x_profile = img[cy, :]
        y_profile = img[:, cx]

        fwhm_x = np.sum(x_profile > (x_profile.max() / 2))
        fwhm_y = np.sum(y_profile > (y_profile.max() / 2))

        # Do not register properties if there is no beam image.
        if not beam_visible(img, cx, cy, droi, threshold):
            continue

        # Dict is indexed by scan number.
        beam_propties[sc] = [xval, [cx, cy], [fwhm_x, fwhm_y]]
        beam_images[sc]   = img

    return beam_images, beam_propties


def beam_visible(img, cx, cy, droi=4, threshold=THRESHOLD):
    """Detect if a beam is present based on the peak-to-average ratio.

    Args:
        img (np.array): The 2D image array to analyze.
        cx (int): The x-coordinate of the beam centroid.
        cy (int): The y-coordinate of the beam centroid.
        droi (int): The half-size of the region around the centroid used
            to calculate the peak intensity. Default is 4 pixels.
        threshold (float): The minimum peak-to-average ratio required to
            consider the beam visible. Default is 10.

    Returns:
        bool: True if the beam is considered visible, False otherwise.
    """
    roi_avg = np.mean(img[cy-droi:cy+droi, cx-droi:cx+droi])
    mean = np.mean(img)
    ratio_rtom = roi_avg / mean if mean != 0 else 0

    return ratio_rtom >= threshold


#
# Methods for reading HDF5 files and exporting to dictionary structure.
#

def data_from_h5_files(files):
    """Read multiple HDF5 files into a nested dict."""
    data = {}
    for filename in files:
        key = re.sub('pass', '',
                     os.path.basename(filename).split('_')[2].split('-')[0])
        data[key] = h5_to_dict(filename)
    return data


def files_in_directory(wdir, pattern):
    """List files in a directory matching a regex pattern."""
    rawfiles = os.listdir(wdir)
    return sorted([f"{wdir}/{f}" for f in rawfiles if re.match(pattern, f)])


def h5_to_dict(filename):
    """Read an HDF5 file into a nested dict.

    Structure:
        data[group_name]['attrs']        -> group attributes (metadata)
        data[group_name][dataset_name]   -> dict with 'data' (numpy array)
                                           and 'attrs' (dataset attributes)
    """
    def _read_group(grp):
        out = {'attrs': dict(grp.attrs)}
        for name, item in grp.items():
            if isinstance(item, h5py.Dataset):
                out[name] = {'data': item[()], 'attrs': dict(item.attrs)}
            elif isinstance(item, h5py.Group):
                out[name] = _read_group(item)
        return out

    with h5py.File(filename, 'r') as f:
        return {key: _read_group(f[key]) for key in f}


#
# Functions to extract device (motor) and observable values across scans.
#

def _get_dev_val(dataset, nscan, dev):
    """Helper function to extract a device value from the dataset."""
    val = dataset[nscan]['attrs'].get(dev)
    if isinstance(val, (np.ndarray, list)):
        val = val[0]
    return val


def get_scan_data(data, variable, observable):
    """Extract observable and variable values across scans."""
    ndset = list(data.keys())
    datascans = []
    for ns in ndset:
        scanlist = list(data[ns].keys())

        obs_set = []
        var_set = []

        for nscan in scanlist:
            obsval = _get_dev_val(data[ns], nscan, observable)
            obs_set.append(obsval)
            varval = _get_dev_val(data[ns], nscan, variable)
            var_set.append(varval)

        datascans.append((obs_set, var_set))

    return datascans


#
# Variable behavior and statistical analysis.
#


def observable_data(data, observable):
    """Extract the behavior of a variable across scans in a dataset.

    Args:
        data (dict): Nested dict containing the set of
            one full scan (one pass).
        observable (str): The variable to extract
            (e.g., 'photocollector', 'centroid').

    Returns:
        tuple: (motor, scans, xval, yval) where:
        motor (str)      : The name of the scanned variable.
        scans (np.array) : Array of scan numbers.
        xval (np.array)  : Array of scanned variable values.
        yval (list of np.array): List of arrays containing the observable
            values for each scan.
    """
    # The scanned variable (idx).
    motor     = data['scan-0000']['attrs']['scan_motor']
    device    = data['scan-0000']['attrs']['scan_type']
    dev_motor = f"{device}.{motor}"

    # Centroids are calculated in beam_centroid().
    if observable == 'centroid':
        scans, xvals, centrs, sigmas = beam_centroid(data, dev_motor)

        # Centroid values, reshaped to [all x, all y].
        centroids = [centrs[:, 0], centrs[:, 1]]

        # Sigmas, reshaped to [all sx, all sy].
        sigmas  = [sigmas[:, 0], sigmas[:, 1]]

        return motor, scans, xvals, centroids, sigmas

    # FWHMs are calculated in beam_fwhm().
    if observable == 'fwhm':
        fwhms = beam_fwhm(data, dev_motor)

        # Dict is ordered by scan number.
        scans = np.array(list(fwhms.keys()))

        # Observable values.
        xvals = np.array([fwhms[sc][0] for sc in scans])

        # FWHM values.
        cvalues = np.array([fwhms[sc][1] for sc in scans])
        fwhms   = [cvalues[:, 0], cvalues[:, 1]]

        return motor, scans, xvals, fwhms, None

    # Intensities are calculated in beam_intensity().
    if observable == 'intensity':
        intensities = beam_intensity(data, dev_motor)

        # Dict is ordered by scan number.
        scans = np.array(list(intensities.keys()))

        # Observable values.
        xvals = np.array([intensities[sc][0] for sc in scans])

        # Intensity values.
        cvalues      = np.array([intensities[sc][1] for sc in scans])
        intensities = [cvalues[:, 0], cvalues[:, 1], cvalues[:, 2]]

        # Sigmas
        sigmas = [np.sqrt(intens) for intens in intensities]

        return motor, scans, xvals, intensities, sigmas

    scans, xval, yval = [], [], []
    # Run over each point scanned.
    for scan, scandata in data.items():
        # Get scan number, scanning index and observable value.
        scans.append(int(scan.split('-')[-1]))
        xmeta = _get_variable_metadata(scandata, dev_motor)
        ymeta = scandata['attrs'].get(f"{device}.{observable}")

        # Append the values, handling both scalar and array metadata cases.
        if isinstance(xmeta, list) or isinstance(xmeta, np.ndarray):
            xval.append(float(xmeta[0]))
        else:
            xval.append(float(xmeta))
        if isinstance(ymeta, list) or isinstance(ymeta, np.ndarray):
            yval.append(float(ymeta[0]))
        else:
            yval.append(float(ymeta))

    return motor, np.array(scans), np.array(xval), [np.array(yval)], None


def observable_statistics(data, observable):
    """Calculate statistics of an observable across scans in a dataset.

    Args:
        data (dict): Nested dict containing the set of all passes.
        observable (str): The variable to analyze (e.g., 'photocollector',
            'centroid').
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.
        threshold (float): The minimum peak-to-average ratio required to
            consider the beam visible. Default is 10.

    Returns:
        dict: A dict mapping dataset keys to statistics of the observable,
            including mean, median, and standard deviation.
    """
    yscans = []
    for dataset in data.values():
        (motor, scans,
         xvals, yvals, sigmas) = observable_data(dataset, observable)
        yscans.append(yvals)

    # Calculate statistics for each dataset.
    yscans = np.array(yscans)
    yavg   = np.mean(yscans, axis=0)
    ymed   = np.median(yscans, axis=0)
    ystd   = np.std(yscans, axis=0)

    stats = {
        'xval'    : xvals,
        'mean'    : yavg,
        'median'  : ymed,
        'std_dev' : ystd
    }
    return stats


#
# Correlation analysis functions.
#

def correlate(a, b):
    """Calculate the normalized cross-correlation between two 1D arrays."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    norm = np.std(a) * np.std(b) * len(a)
    return np.correlate(a, b, mode='full') / norm


#
# Plotting functions.
#

def centroid_plot(data, scanpass, wdir='.', save_fmt='gif'):
    """Plot the beam profile images and their centroids in a dataset.

    Args:
        data (dict): Nested dict containing the set of one full scan
            (one pass).
        scanpass (str): Identifier for the scan pass
            (used in titles and filenames).
        wdir (str): Working directory to save the plots.
            Default is current directory.
        save_fmt (str): Format to save the animation
            ('gif', 'mp4', or '' to skip saving).
    """
    # Use the existing function to get motor, positions and centroids.
    (motor, scan_nums, xval,
     (cx_all, cy_all)) = observable_data(data, 'centroid')

    # Retrieve images in the same order as scan_nums.
    images = [(f'scan-{sc:04d}',
               data[f'scan-{sc:04d}']['dvf_B1']['data'])
               for sc in scan_nums]

    fig, (ax_img, ax_cx, ax_cy) = plt.subplots(1, 3, figsize=(24, 5))

    # Left: image
    im = ax_img.imshow(images[0][1], cmap='viridis', animated=True)
    plt.colorbar(im, ax=ax_img, label='Intensity')
    ax_img.set_xlabel('Pixel X')
    ax_img.set_ylabel('Pixel Y')
    img_title = ax_img.set_title('')

    # Center: centroid X trace with a moving marker
    ax_cx.plot(xval, cx_all, marker='o',
            color='steelblue', label='Centroid X')
    marker_pt, = ax_cx.plot([], [], 'ro',
                            markersize=10, label='Current')
    ax_cx.set_xlabel(motor.capitalize())
    ax_cx.set_ylabel('Centroid X (px)')
    ax_cx.set_title('Centroid X')
    ax_cx.legend()
    ax_cx.grid(True)

    # Right: centroid Y trace with a moving marker
    ax_cy.plot(xval, cy_all, marker='o',
               color='orange', label='Centroid Y')
    marker_pt_y, = ax_cy.plot([], [], 'ro',
                              markersize=10, label='Current')
    ax_cy.set_xlabel(motor.capitalize())
    ax_cy.set_ylabel('Centroid Y (px)')
    ax_cy.set_title('Centroid Y')
    ax_cy.legend()
    ax_cy.grid(True)

    def update(frame):
        name, img = images[frame]
        im.set_data(img)
        im.set_clim(img.min(), img.max())
        img_title.set_text(f'Scan: {name}')
        marker_pt.set_data([xval[frame]], [cx_all[frame]])
        marker_pt_y.set_data([xval[frame]], [cy_all[frame]])
        return im, img_title, marker_pt, marker_pt_y

    anim = FuncAnimation(fig, update, frames=len(images),
                         interval=300, blit=True)
    plt.tight_layout(pad=3.0)

    if save_fmt == 'gif':
        anim.save(f'{wdir}/beam_xy_pass_{scanpass}.gif',
                writer='pillow', fps=2, dpi=300)
    elif save_fmt == 'mp4':
        # Alternatively, save as mp4 (requires ffmpeg):
        anim.save(f'{wdir}/beam_xy_pass_{scanpass}.mp4',
                writer='ffmpeg', fps=2, dpi=300)
    else:
        print("File format not specified or unknown"
              " (use 'gif' or 'mp4' to save to file)."
              " Skipping saving process.")
        pass

    plt.close()
    ipydisplay(HTML(anim.to_jshtml()))


def centroid_x_delta_plot(dataset, motor, scan_start=0, scan_end=-1):
    """Plot motor change and centroid X change across scan passes.

    Args:
        dataset (dict): Nested dict containing all scan passes.
        motor (str): Motor observable to plot, e.g. 'mirror.cs_rz'
            or 'mirror.ry'.
        scan_start (int): Scan index for the initial centroid measurement.
        scan_end (int): Scan index for the final centroid measurement.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    rax0 = axs[0].twinx()
    rax1 = axs[1].twinx()
    plt.subplots_adjust(hspace=0.3)

    baseline_motor = []
    final_motor    = []
    scan_idx       = []
    baseline_cx    = []
    final_cx       = []

    # Get beam info for all scans in dataset.

    for data in dataset.values():
        baseline_motor.append(
            data[f'scan-{scan_start:04d}']['attrs'][motor][0]
            )
        final_motor.append(
            data[f'scan-{scan_end:04d}']['attrs'][motor][0]
            )
        scan_idx.append(len(scan_idx))

        scans, xvals, centroid, sigmas = beam_centroid(data, motor)
        baseline_cx.append(centroid[scan_start][0])
        final_cx.append(centroid[scan_end][0])

    init_motor = baseline_motor[0]
    init_cx    = baseline_cx[0]

    baseline_motor = np.array(baseline_motor) - init_motor
    final_motor    = np.array(final_motor) - init_motor
    baseline_cx    = np.array(baseline_cx) - init_cx
    final_cx       = np.array(final_cx) - init_cx

    motor_delta = final_motor - baseline_motor
    cx_delta    = final_cx - baseline_cx

    motor_cumulative = np.cumsum(motor_delta)
    cx_cumulative    = np.cumsum(cx_delta)

    motor_label = motor.replace('.', ' ').upper()
    title = f'{motor_label} Change vs. Scan Index'
    left_label = f'{motor_label} Change'

    line0,  = axs[0].plot(scan_idx, motor_delta, marker='o', color='blue',
                          label=motor_label)
    line0b, = rax0.plot(scan_idx, cx_delta, marker='o', color='green',
                       label='Centroid X Change (px)')

    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = rax0.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2, loc='best')

    axs[0].grid(True)
    axs[0].set_xlabel('Scan Index')
    axs[0].set_ylabel(left_label)
    rax0.set_ylabel('Centroid X Change (px)')
    axs[0].set_title(title)

    axs[1].plot(scan_idx, motor_cumulative, marker='o', color='blue',
                label=f'Cumulative {motor_label} Change')
    rax1.plot(scan_idx, cx_cumulative, marker='o', color='green',
              label='Cumulative Centroid X Change')

    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = rax1.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2, loc='best')

    axs[1].grid(True)
    axs[1].set_xlabel('Scan Index')
    axs[1].set_ylabel(f'Cumulative {motor_label} Change')
    rax1.set_ylabel('Cumulative Centroid X Change (px)')
    axs[1].set_title(f'Cumulative {motor_label} Change vs. Scan Index')

    plt.show()


def dataset_plot(ax, xvals, yvals, datakey, observable, motor,
                 first_item=0, last_item=None, annotate_points=True):
    """Plot the behavior of an observable across scans for a single dataset."""
    # Plot the observable vs. scan number for the given dataset.
    # for yval in yvals:

    if annotate_points:
        for i, (tx, yv) in enumerate(zip(xvals[first_item:last_item],
                                         yvals[first_item:last_item])):
            ax.annotate(str(i), (tx, yv), textcoords='offset points',
                         xytext=(5, 5), fontsize=8)
    if last_item is None:
        ax.plot(xvals[first_item:], yvals[first_item:],
                marker='o', label=f'Dataset {datakey}')
    else:
        ax.plot(xvals[first_item:last_item],
                yvals[first_item:last_item],
                marker='o', label=f'Dataset {datakey}')

    ax.set_xlabel(motor)
    ax.set_ylabel(observable)
    ax.set_title(f'{observable.capitalize()} vs. {motor.capitalize()}')
    ax.legend()
    ax.grid(True)


def fwhm_plot(dataset, scanpass, wdir='.', save_fmt='gif'):
    """Plot the beam profile images and their FWHMs in a dataset.

    Args:
        dataset (dict): Nested dict containing the set of one full scan
            (one pass).
        scanpass (str): Identifier for the scan pass
            (used in titles and filenames).
        wdir (str): Working directory to save the plots.
            Default is current directory.
        save_fmt (str): Format to save the animation
            ('gif', 'mp4', or '' to skip saving).
    """
    # Use the existing function to get motor, positions and centroids.
    (motor, scan_nums, xval,
     (fx_all, fy_all)) = observable_data(dataset, 'fwhm')

    # Retrieve images in the same order as scan_nums.
    images = [(f'scan-{sc:04d}',
            dataset[f'scan-{sc:04d}']['dvf_B1']['data'])
            for sc in scan_nums]

    fig, (ax_img, ax_fx, ax_fy) = plt.subplots(1, 3, figsize=(24, 5))

    # Left: image
    im = ax_img.imshow(images[0][1], cmap='viridis', animated=True)
    plt.colorbar(im, ax=ax_img, label='Intensity')
    ax_img.set_xlabel('Pixel X')
    ax_img.set_ylabel('Pixel Y')
    img_title = ax_img.set_title('')

    # Center: centroid X trace with a moving marker
    ax_fx.plot(xval, fx_all, marker='o',
            color='steelblue', label='FWHM X')
    marker_pt, = ax_fx.plot([], [], 'ro',
                            markersize=10, label='Current')
    ax_fx.set_xlabel(motor.capitalize())
    ax_fx.set_ylabel('FWHM X (px)')
    ax_fx.set_title('FWHM X')
    ax_fx.legend()
    ax_fx.grid(True)

    # Right: centroid Y trace with a moving marker
    ax_fy.plot(xval, fy_all, marker='o',
               color='orange', label='FWHM Y')
    marker_pt_y, = ax_fy.plot([], [], 'ro',
                              markersize=10, label='Current')
    ax_fy.set_xlabel(motor.capitalize())
    ax_fy.set_ylabel('FWHM Y (px)')
    ax_fy.set_title('FWHM Y')
    ax_fy.legend()
    ax_fy.grid(True)

    def update(frame):
        name, img = images[frame]
        im.set_data(img)
        im.set_clim(img.min(), img.max())
        img_title.set_text(f'Scan: {name}')
        marker_pt.set_data([xval[frame]], [fx_all[frame]])
        marker_pt_y.set_data([xval[frame]], [fy_all[frame]])
        return im, img_title, marker_pt, marker_pt_y

    anim = FuncAnimation(fig, update, frames=len(images),
                         interval=300, blit=True)
    plt.tight_layout(pad=3.0)

    if save_fmt == 'gif':
        anim.save(f'{wdir}/beam_xy_pass_{scanpass}_fwhm.gif',
                writer='pillow', fps=2, dpi=300)
    elif save_fmt == 'mp4':
        # Alternatively, save as mp4 (requires ffmpeg):
        anim.save(f'{wdir}/beam_xy_pass_{scanpass}_fwhm.mp4',
                writer='ffmpeg', fps=2, dpi=300)
    else:
        print("File format not specified or unknown"
              " (use 'gif' or 'mp4' to save to file)."
              " Skipping saving process.")
        pass

    plt.close()
    ipydisplay(HTML(anim.to_jshtml()))


def plot_double_observable(axs, nrow, dataset, observable, observables,
                           first_item=0, last_item=None):
    """Plot two-component observables in separate subplots."""
    for key, data in dataset.items():
        (motor, scans,
         xvals, yvals, sigmas) = observable_data(data, observable)
        dataset_plot(axs[nrow, 0], xvals, yvals[0], key,
                     f"{observable} X", motor, first_item, last_item)
        dataset_plot(axs[nrow, 1], xvals, yvals[1], key,
                     f'{observable} Y', motor, first_item, last_item)
    observables.remove(observable)
    return


def scan_plot(data, observables, first_item=0, last_item=None):
    """Plot the behavior of an observable across scans for each dataset.

    Args:
        data (dict): Nested dict containing the set of all passes.
        observables (list of str): The variables to plot
            (e.g., 'photocollector', 'centroid').
        first_item (int): The first point of the scan to be included in
            the plot. It allows skipping initial points if desired.
            Default is 0 (include all points).
        last_item (int or None): The last point of the scan to be included
            in the plot. If None, include all points.
    """
    # Determine the number of rows and columns for subplots based on
    # the number of observables.
    nobs = len(observables)
    # Add an extra plot for the centroid components.
    nobs += sum([1 for obs in ['centroid', 'fwhm', 'intensity']
                 if obs in observables])

    nrows = max((nobs + 1) // 2, 1)
    ncols = 2 if nobs > 1 else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(10 * ncols, 6 * nrows))
    # Make it iterable
    if nrows == 1 and ncols == 1:
        axs = [axs]
    # elif nrows == 1 or ncols == 1:
    #     axs = axs.flatten()

    # Some observables demand two subplots.
    nextrow = 0
    for observable in ['centroid', 'fwhm', 'intensity']:
        if observable in observables:
            plot_double_observable(axs, nextrow, data,
                                   observable, observables,
                                   first_item, last_item)
            nextrow += 1

    # Loop over each observable and dataset to plot the
    # observable vs. scan number.
    for idx, observable in enumerate(observables):
        nr, nc = divmod(idx + nextrow * ncols, 2)
        ax = axs[nr, nc] if nrows > 1 and ncols > 1 else axs[idx + nextrow]
        for key, dataset in data.items():
            (motor, scans,
             xvals, yvals, sigmas) = observable_data(dataset, observable)
            for yval in yvals:
                dataset_plot(ax, xvals, yval, key, observable, motor,
                            first_item, last_item)

    plt.show()


# Old code for caustic analysis. To be checked.

def caustic_analysis(filename, filedir):
    """Perform caustic analysis on the given scan data."""
    file = '/'.join([filedir, filename])

    f = h5py.File(name=file, mode='r')

    positions = [f[scaname].attrs['z_pos'] for scaname in f]

    caustic3d = []
    for scaname in f:
        img = np.array(f[scaname]['dvf2'])
        caustic3d.append(img)

    causticx = np.sum(caustic3d, axis=1).T
    causticy = np.sum(caustic3d, axis=2).T

    fwhmsx = [
        utils.full_width(data=profilex, coords=positions)[0]
        for profilex in causticx.T]
    fwhmsy = [
        utils.full_width(data=profiley, coords=positions)[0]
        for profiley in causticy.T]

    causticx_params, covx = curve_fit(utils.caustic_func,
                                      positions,
                                      fwhmsx,
                                      p0=None)
    causticy_params, covy = curve_fit(utils.caustic_func,
                                      positions,
                                      fwhmsy,
                                      p0=None)

    resultsx = utils.caustic_processing(causticx_params, positions)
    resultsy = utils.caustic_processing(causticy_params, positions)

    return resultsx, resultsy
