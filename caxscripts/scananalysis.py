"""Set of methods to analyze scan data and extract beam properties.

This module provides functions to analyze CARCARÁ-X beamline data obtained
from scanning with its motors. All data is expected to be stored in HDF5 files.
Functions are designed to read data, extract beam properties (centroids,
intensities, FWHMs), and visualize results. The main focus is analyzing
beam profiles and their variation with respect to the scanned variable
(e.g., variations in Tx).

Data Levels:
The analysis operates at three hierarchical data levels:
- data:     A single scan step, containing raw image data and metadata
        for that step (e.g., one DVF image and motor position).
- scan:     A full scan (one pass), composed of multiple 'data' steps
        (e.g., all steps from one HDF5 file).
- dataset:  Multiple scan passes of the same type, containing multiple
        'scandata' dicts (e.g., all HDF5 files from repeated scans).

Function Categories (matching module section structure):

Scan parameter extraction methods:
- _get_variable_values: Extract scanned variable values from a single
        data step (data level).

Beam properties extraction methods and analysis:
- beam_from_scan: Extract beam properties from all steps in a scan
        (scandata level).
- beam_centroid:  Return centroids for all steps in a scan (scandata level;
        wraps beam_from_scan).
- beam_fwhm:      Return FWHMs for all steps in a scan (scandata level; wraps
        beam_from_scan).
- beam_intensity: Return total intensity for all steps in a scan
        (scandata level; wraps beam_from_scan).

Methods for reading HDF5 files and exporting to dictionary structure:
- files_in_directory: List files in a directory matching a regex pattern
                      (utility, no scan data input).
- h5_to_dict:        Read a single HDF5 file into a scandata dict
        (scandata level).
- dataset_from_h5_files: Aggregate multiple HDF5 files into a dataset dict
        (dataset level).

Functions to extract device (motor) and observable values across scans:
- _get_dev_val:      Helper to extract device values from a scandata step
        (scandata level).
- get_scan_data:     Extract observable and variable values across all
        dataset passes (dataset level).

Variable behavior and statistical analysis:
- observable_data:        Extract observable behavior across steps in a single
        scan (scandata level).
- observable_statistics:   Calculate mean/median/std of an observable across
        dataset passes (dataset level).

Correlation analysis functions:
- correlate: Calculate normalized cross-correlation between two 1D arrays
        (utility, generic math).

Plotting functions (see dedicated section below):
- dataset_plot:  Plot observable trace for a single scandata (scandata level).
- centroid_plot: Animate beam images and centroids for a scan pass
        (scandata level).
- fwhm_plot:     Animate beam images and FWHMs for a scan pass
        (scandata level).
- scan_plot:     Main entry point to plot all observables across dataset
        passes (dataset level).
- centroid_x_delta_plot:  Plot motor/centroid changes across dataset passes
        (dataset level).
- plot_double_observable: Plot two-component observables across passes
        (dataset level; helper).

Legacy code (to be checked):
- caustic_analysis: Perform caustic analysis on a single HDF5 file
        (scandata level).
"""
import h5py
import numpy as np
import os
import re

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from IPython.display import HTML
from IPython.display import display as ipydisplay

# from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from . import utils

from caxscripts.image_statistics import Histogram2DAnalyzer

# Threshold for peak-to-average ratio acceptance of image.
THRESHOLD = 100


# ==================================================
#                    Utilities
# ==================================================

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
# Correlation analysis functions.
#
def correlate(a, b):
    """Calculate the normalized cross-correlation between two 1D arrays."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    norm = np.std(a) * np.std(b) * len(a)
    return np.correlate(a, b, mode='full') / norm


# ==================================================
#           Dataset level (multiple passes)
# ==================================================

def dataset_from_h5_files(files):
    """Read multiple HDF5 files into a `dataset` nested dict."""
    dataset = {}
    for filename in files:
        key = re.sub('pass', '',
                     os.path.basename(filename).split('-')[0].split('_')[-1])
        dataset[key] = h5_to_dict(filename)
    return dataset


def _get_dev_val(dataset, nscan, dev):
    """Helper function to extract a device value from the dataset."""
    val = dataset[nscan]['attrs'].get(dev)
    if isinstance(val, (np.ndarray, list)):
        val = val[0]
    return val


# Not used at the moment?
def get_scan_data(dataset, variable, observable):
    """Extract observable and variable values across scans."""
    ndset = list(dataset.keys())
    scandata = []
    for ns in ndset:
        scanlist = list(dataset[ns].keys())

        obs_set = []
        var_set = []

        for nscan in scanlist:
            obsval = _get_dev_val(dataset[ns], nscan, observable)
            obs_set.append(obsval)
            varval = _get_dev_val(dataset[ns], nscan, variable)
            var_set.append(varval)

        scandata.append((obs_set, var_set))

    return scandata


#
# Variable behavior and statistical analysis.
#
def observable_statistics(dataset, observable):
    """Calculate statistics of an observable across scans in a dataset.

    Args:
        dataset (dict): Nested dict containing the set of all passes.
        observable (str): The variable to analyze (e.g., 'photocollector',
            'centroid').
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.

    Returns:
        dict: A dict mapping dataset keys to statistics of the observable,
            including mean, median, and standard deviation.
    """
    yscans = []
    for scandata in dataset.values():
        (motor, scans,
         xvals, yvals, sigmas) = observable_data(scandata, observable)
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


# ==================================================
#           Scan level (one full scan)
# ==================================================

def _get_scan_names(datastep):
    attrs0    = datastep['attrs']
    scanname  = attrs0['scan_name']
    scandev   = attrs0['scan_device']
    scanmotor = attrs0.get('scan_motor')
    dev_motor = f"{scandev}"
    if scanmotor is not None:
        dev_motor += f".{scanmotor}"
    return scanname, scandev, scanmotor, dev_motor


def observable_data(datascan, observable, droi=4):
    """Extract the behavior of a variable across steps in a scan.

    Args:
        datascan (dict): Nested dict containing the set of
                         one full scan (one pass).
        observable (str): The variable to extract (e.g.,
                          'photocollector', 'centroid').
        droi (int): central region around the centroid to estimate intensity
                        by averaging the peak.

    Returns:
        tuple: (motor, steps, xval, yval) where:
        motor (str)      : The name of the scanned variable.
        steps (np.array) : Array of step numbers.
        xval (np.array)  : Array of scanned variable values.
        yval (list of np.array): List of arrays containing the observable
                                 values for each scan.
    """
    # The scanned variable name (device.motor).
    dev_motor = _get_scan_names(datascan['scan-0000'])[3]

    # Get observable instances.
    if observable in ['centroid', 'fwhm', 'intensity']:
        beam_instances = beam_from_scan(datascan, droi)

    # Centroids are calculated in beam_centroid().
    if observable == 'centroid':
        steps, xvals, centrs, sigmas = beam_centroid(beam_instances)

        # Centroid values, reshaped to [all x, all y].
        centroids = [centrs[:, 0], centrs[:, 1]]

        # Sigmas, reshaped to [all sx, all sy].
        sigmas  = [sigmas[:, 0], sigmas[:, 1]]

        return dev_motor, steps, xvals, centroids, sigmas

    # FWHMs are calculated in beam_fwhm().
    if observable == 'fwhm':
        fwhms = beam_fwhm(beam_instances)

        # Dict is ordered by scan number.
        steps = np.array(list(fwhms.keys()))

        # Observable values.
        xvals = np.array([fwhms[step][0] for step in steps])

        # FWHM values.
        cvalues = np.array([fwhms[step][1] for step in steps])
        fwhms   = [cvalues[:, 0], cvalues[:, 1]]

        return dev_motor, steps, xvals, fwhms, None

    # Intensities are calculated in beam_intensity().
    if observable == 'intensity':
        expo_time = datascan['scan-0000']['dvf_B1']['attrs']['expo_time']
        xvals, intensities = beam_intensity(beam_instances,
                                            droi=droi, expo_time=expo_time)

        # Dict is ordered by scan number.
        steps = np.arange(len(xvals))

        # Intensity values.
        # cvalues     = np.array([intensities[step] for step in steps])
        # intensities = [cvalues[:, 0], cvalues[:, 1], cvalues[:, 2]]

        # Sigmas
        sigmas = [np.sqrt(intens) for intens in intensities]

        return dev_motor, steps, xvals, intensities, sigmas

    steps, xval, yval = [], [], []
    # Run over each point scanned.
    for step, stepdata in datascan.items():
        # Get scan number, scanning index and observable value.
        steps.append(int(step.split('-')[-1]))
        xdata = _get_variable_values(stepdata, dev_motor)

        # Seach for observable within data attribute list.
        for atr in stepdata['attrs'].keys():
            if re.search(observable, atr):
                ydata = stepdata['attrs'].get(atr)
                break

        # Append the values, handling both scalar and array metadata cases.
        if xdata.ndim == 0:
            xval.append(float(xdata))
        elif xdata.ndim == 1:
            xval.append(xdata[0])
        else:
            # Take only the values, not the limits. This is the slits case,
            # so the four values of the blades must be transformed into
            # the coordinates of the center of the slit.
            xval.append(xdata[:, 0])

        if isinstance(ydata, np.ndarray):
            yval.append(float(ydata[0]))
        else:
            yval.append(float(ydata))

    return dev_motor, np.array(steps), np.array(xval), [np.array(yval)], None


#
# Beam properties extraction methods and analysis.
#
def beam_from_scan(datascan, droi=4, analysis_mode='qck'):
    """Return properties of the beam profiles from the datascan dict.

    Operates on a full scan, linking scanned variable value to beam
    images and properties. Utilizes the `Histogram2DAnalyzer` class
    from the `image_statistics` module.

    Args:
        datascan (dict): Nested dict containing the set of one full scan.
        droi (int): The half-size of the region around the centroid used to
            determine if the beam is visible. Default is 4 pixels.
        analysis_mode (str): what method to use for analyzing image properties,
            'qck' for quick analysis, 'mom' for analysis through moments
            calculation and 'fit' for non-linear gaussian fitting. Defaults
            to 'qck'.

    Returns:
        beam_instances (dict): A dict mapping step numbers to H2DA instances
            containing the image and its properties, only scans where the beam
            is visible are returned.
    """
    beam_instances = {}

    for step, dataset in datascan.items():
        # Extract scanning index and observable value.
        st = int(step.split('-')[-1])

        # Extract the scanned variable value for this step.
        values = _get_variable_values(dataset)

        if values.ndim == 0:
            xvals = float(values)
        elif values.ndim == 1:
            xvals = values[0]
        else:
            xvals = values[:, 0]

        # Get image data and calculate centroid.
        img = dataset['dvf_B1']['data'].T
        img_xedges = np.arange(img.shape[0]+1)
        img_yedges = np.arange(img.shape[1]+1)

        ana = Histogram2DAnalyzer(img,
                                  droi=droi,
                                  x_bin_edges=img_xedges,
                                  y_bin_edges=img_yedges)

        if not ana.beam_visible:
            continue

        # Perform the analysis to extract beam properties based on the
        # specified mode. Quick analysis is always perfomed
        ana.analyze(analysis_mode)

        # Return a dict linking scanned variable value to a H2DA
        # instance containing the image and its properties.
        beam_instances[st] = [xvals, ana]

        # Right now, as 'qck' is set as the default analysis mode, the
        # behaviour is the exact same as was previously implemented,
        # justifying the deletion of the old code. For moment and fitting
        # calculation, it remains to define the use of a ROI and to handle
        # thresholding adequately.

    return beam_instances


def beam_centroid(beam_instances, analysis_mode='qck'):
    """Return centroids of the beam profiles from the data dict.

    Args:
        beam_instances (dict): A dict mapping step numbers to H2DA instances
            containing the image and its properties.
        analysis_mode (str): The mode of analysis to use. Default is 'qck'.

    Returns:
        dict: A dict mapping scan numbers to (cx, cy) centroids.
    """
    steps, coordinates, centrs, sigmas  = [], [], [], []
    # for step, values in beam_instances.items():
    for step, (scanval, ana) in beam_instances.items():
        steps.append(step)
        sv = np.asarray(scanval)
        if sv.ndim == 0:
            coordinates.append(float(sv))
        else:
            # Take only the values, not the limits. This is the slits case,
            # so the four values of the blades must be transformed into
            # the coordinates of the center of the slit.
            coordinates.append(sv[:, 0])

        hprm = getattr(ana, f"hprm_{analysis_mode}")
        centrs.append([hprm['mux'], hprm['muy']])
        sigmas.append([hprm['sigx'], hprm['sigy']])

    return (np.array(steps), np.array(coordinates),
            np.array(centrs), np.array(sigmas))


def beam_fwhm(beam_instances, analysis_mode='qck'):
    """Return fwhm of the beam profiles from the data dict.

    Args:
        beam_instances (dict): A dict mapping step numbers to H2DA instances
            containing the image and its properties.
        analysis_mode (str): The mode of analysis to use. Default is 'qck'.

    Returns:
        fwhms (dict): A dict mapping scan numbers to (fx, fy) fwhm's.
    """
    fwhms = {}
    for st in beam_instances.keys():
        xval = beam_instances[st][0]

        ana = beam_instances[st][1]
        hprm = getattr(ana, f"hprm_{analysis_mode}")
        fx, fy = hprm['fwhmx'], hprm['fwhmy']

        fwhms[st] = [xval, [fx, fy]]

    return fwhms


def beam_intensity(beam_instances, droi=8, expo_time=0.1, analysis_mode='qck'):
    """Return the total intensity of the beam profiles from the data dict.

    Args:
        beam_instances (dict): A dict mapping step numbers to H2DA instances
            containing the image and its properties.
        droi (int): region around the centroid to estimate intensity by
                 averaging the peak.
        expo_time (float): Exposure time of the image. Default is 0.1 seconds.
        analysis_mode (str): The mode of analysis to use. Default is 'qck'.

    Returns:
        dict: A dict mapping scan numbers to total intensity.
    """
    xvals = []
    peaks, intens_by_mask, peaks_by_fwhm = [], [], []
    for step in beam_instances.keys():
        xval = beam_instances[step][0]
        ana  = beam_instances[step][1]
        hprm = getattr(ana, f"hprm_{analysis_mode}")

        if not ana.beam_visible:
            continue

        img            = ana.img
        cx, cy         = hprm['mux'], hprm['muy']
        fwhm_x, fwhm_y = hprm['fwhmx'], hprm['fwhmy']

        peak = np.mean(img[cx - droi : cx + droi + 1,
                           cy - droi : cy + droi + 1])

        mask = img > (peak / 2)
        area_mask  = np.sum(mask)
        img_masked = np.where(mask, img, 0)
        area_img_masked = np.sum(img_masked)

        intensity_by_mask = (area_img_masked / (area_mask * expo_time)
                             if area_mask != 0 else 0)

        peak /= expo_time
        peak_fwhm_norm = (peak / (fwhm_x * fwhm_y)
                          if fwhm_x * fwhm_y != 0 else 0)

        xvals.append(xval)
        peaks.append(peak)
        intens_by_mask.append(intensity_by_mask)
        peaks_by_fwhm.append(peak_fwhm_norm)
        # intensities[step] = [peak, intensity_by_mask, peak_fwhm_norm]

    intensities = [
        np.array(peaks),
        np.array(intens_by_mask),
        np.array(peaks_by_fwhm)
    ]

    return np.array(xvals), intensities


# ==================================================
#           Data level (one scan step)
# ==================================================

#
# Scan parameter extraction methods.
#
def _get_variable_values(data, dev_motor=None):
    """Helper function to extract the variable metadata from the dataset.

    Args:
        data (dict): Dict containing data for a single step of a scan.
        dev_motor (str): The device and motor being scanned
            (e.g., 'mirror.rx').

    Returns:
        np.ndarray: values of the scanned variable,
             [value, lolm, hilm, enable].
    """
    # Separate device and motor from the input string.

    # Get scan name.
    sname, sdev, smotor, dev_motor = _get_scan_names(data)

    # Check for slits case.
    if 'slit' in sname:
        vals = []
        for side in ['top', 'right', 'bottom', 'left']:
            vals.append(data['attrs'].get(f"{sdev}.{side}"))
        return np.array(vals)

    # Motor is defined in other cases.
    scanmotor = data['attrs'].get('scan_motor')
    dev_motor = f"{sdev}.{scanmotor}"
    return np.array(data['attrs'].get(dev_motor))


# ==================================================
#              Plotting functions
# ==================================================
# Plotting functions are organized by scope:
# - Scan-level animations: centroid_plot, fwhm_plot
# - Dataset-level plots: centroid_x_delta_plot, dataset_plot,
#   plot_double_observable, scan_plot
# - Utility helpers: (none at this time)
#

def centroid_plot(data, steppass, wdir='.', save_fmt='gif'):
    """Plot the beam profile images and their centroids in a dataset.

    Args:
        data (dict): Nested dict containing the set of one full scan
            (one pass).
        steppass (str): Identifier for the scan pass
            (used in titles and filenames).
        wdir (str): Working directory to save the plots.
            Default is current directory.
        save_fmt (str): Format to save the animation
            ('gif', 'mp4', or '' to skip saving).
    """
    # Use the existing function to get motor, positions and centroids.
    (motor, steps, xval,
     (cx_all, cy_all), _) = observable_data(data, 'centroid')

    # Retrieve images in the same order as step_nums.
    images = [(f'step-{step:04d}',
               data[f'step-{step:04d}']['dvf_B1']['data'])
               for step in steps]

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
        img_title.set_text(f'Step: {name}')
        marker_pt.set_data([xval[frame]], [cx_all[frame]])
        marker_pt_y.set_data([xval[frame]], [cy_all[frame]])
        return im, img_title, marker_pt, marker_pt_y

    anim = FuncAnimation(fig, update, frames=len(images),
                         interval=300, blit=True)
    plt.tight_layout(pad=3.0)

    if save_fmt == 'gif':
        anim.save(f'{wdir}/beam_xy_pass_{steppass}.gif',
                writer='pillow', fps=2, dpi=300)
    elif save_fmt == 'mp4':
        # Alternatively, save as mp4 (requires ffmpeg):
        anim.save(f'{wdir}/beam_xy_pass_{steppass}.mp4',
                writer='ffmpeg', fps=2, dpi=300)
    else:
        print("File format not specified or unknown"
              " (use 'gif' or 'mp4' to save to file)."
              " Skipping saving process.")
        pass

    plt.close()
    ipydisplay(HTML(anim.to_jshtml()))


def centroid_x_delta_plot(dataset, motor, step_start=0, step_end=-1):
    """Plot motor change and centroid X change across step passes.

    Args:
        dataset (dict): Nested dict containing all step passes.
        motor (str): Motor observable to plot, e.g. 'mirror.cs_rz'
            or 'mirror.ry'.
        step_start (int): Step index for the initial centroid measurement.
        step_end (int): Step index for the final centroid measurement.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    rax0 = axs[0].twinx()
    rax1 = axs[1].twinx()

    baseline_motor = []
    final_motor    = []
    step_idx       = []
    baseline_cx    = []
    final_cx       = []

    # Get beam info for all steps in dataset.

    for data in dataset.values():
        baseline_motor.append(
            data[f'step-{step_start:04d}']['attrs'][motor][0]
            )
        final_motor.append(
            data[f'step-{step_end:04d}']['attrs'][motor][0]
            )
        step_idx.append(len(step_idx))

        beam_instances = beam_from_scan(data)
        steps, xvals, centroid, sigmas = beam_centroid(beam_instances)
        baseline_cx.append(centroid[step_start][0])
        final_cx.append(centroid[step_end][0])

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

    line0,  = axs[0].plot(step_idx, motor_delta, marker='o', color='blue',
                          label=motor_label)
    line0b, = rax0.plot(step_idx, cx_delta, marker='o', color='green',
                       label='Centroid X Change (px)')

    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = rax0.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2, loc='best')

    axs[0].grid(True)
    axs[0].set_xlabel('Step Index')
    axs[0].set_ylabel(left_label)
    rax0.set_ylabel('Centroid X Change (px)')
    axs[0].set_title(title)

    axs[1].plot(step_idx, motor_cumulative, marker='o', color='blue',
                label=f'Cumulative {motor_label} Change')
    rax1.plot(step_idx, cx_cumulative, marker='o', color='green',
              label='Cumulative Centroid X Change')

    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = rax1.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2, loc='best')

    axs[1].grid(True)
    axs[1].set_xlabel('Step Index')
    axs[1].set_ylabel(f'Cumulative {motor_label} Change')
    rax1.set_ylabel('Cumulative Centroid X Change (px)')
    axs[1].set_title(f'Cumulative {motor_label} Change vs. Step Index')

    plt.show()


def dataset_plot(ax, xvals, yvals, datakey, observable, motor,
                 first_item=0, last_item=None, annotate_points=True):
    """Plot the behavior of an observable across steps for a single dataset."""
    # Plot the observable vs. step number for the given dataset.
    # for yval in yvals:
    if annotate_points:
        for i, (tx, yv) in enumerate(
            zip(xvals[first_item:last_item],  # noqa: B905
            yvals[first_item:last_item])      # noqa: B905
            ):
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


def fwhm_plot(dataset, steppass, wdir='.', save_fmt='gif'):
    """Plot the beam profile images and their FWHMs in a dataset.

    Args:
        dataset (dict): Nested dict containing the set of one full scan
            (one pass).
        steppass (str): Identifier for the step pass
            (used in titles and filenames).
        wdir (str): Working directory to save the plots.
            Default is current directory.
        save_fmt (str): Format to save the animation
            ('gif', 'mp4', or '' to skip saving).
    """
    # Use the existing function to get motor, positions and centroids.
    (motor, step_nums, xval,
     (fx_all, fy_all)) = observable_data(dataset, 'fwhm')

    # Retrieve images in the same order as step_nums.
    images = [(f'step-{step:04d}',
               dataset[f'step-{step:04d}']['dvf_B1']['data'])
               for step in step_nums]

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
        img_title.set_text(f'Step: {name}')
        marker_pt.set_data([xval[frame]], [fx_all[frame]])
        marker_pt_y.set_data([xval[frame]], [fy_all[frame]])
        return im, img_title, marker_pt, marker_pt_y

    anim = FuncAnimation(fig, update, frames=len(images),
                         interval=300, blit=True)
    plt.tight_layout(pad=3.0)

    if save_fmt == 'gif':
        anim.save(f'{wdir}/beam_xy_pass_{steppass}_fwhm.gif',
                writer='pillow', fps=2, dpi=300)
    elif save_fmt == 'mp4':
        # Alternatively, save as mp4 (requires ffmpeg):
        anim.save(f'{wdir}/beam_xy_pass_{steppass}_fwhm.mp4',
                writer='ffmpeg', fps=2, dpi=300)
    else:
        print("File format not specified or unknown"
              " (use 'gif' or 'mp4' to save to file)."
              " Skipping saving process.")
        pass

    plt.close()
    ipydisplay(HTML(anim.to_jshtml()))


def plot_double_observable(axs, nextaxis, dataset, observable,
                           first_item=0, last_item=None, droi=4):
    """Plot two-component observables in separate subplots."""
    for key, data in dataset.items():
        (motor, steps,
         xvals, yvals, sigmas) = observable_data(data, observable, droi=droi)
        dataset_plot(axs[nextaxis], xvals, yvals[0], key,
                     f"{observable} X", motor, first_item, last_item)
        dataset_plot(axs[nextaxis+1], xvals, yvals[1], key,
                     f'{observable} Y', motor, first_item, last_item)


def scan_plot(dataset, observables, first_item=0, last_item=None, droi=8):
    """Plot the behavior of an observable across scans for each dataset.

    Args:
        dataset (dict): Nested dict containing the set of all passes.
        observables (list of str): The variables to plot
            (e.g., 'photocollector', 'centroid').
        first_item (int): The first point of the scan to be included in
            the plot. It allows skipping initial points if desired.
            Default is 0 (include all points).
        last_item (int or None): The last point of the scan to be included
            in the plot. If None, include all points.
        droi (int): central region defined by the user to average on.
    """
    # Determine the number of rows and columns for subplots based on
    # the number of observables.
    nobs = len(observables)
    # Add an extra plot for the centroid components.
    nobs += sum([1 for obs in ['centroid', 'fwhm']
                 if obs in observables])
    if 'intensity' in observables:
        nobs += 2

    nrows = max((nobs + 1) // 2, 1)
    ncols = 2 if nobs > 1 else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(10 * ncols, 6 * nrows),
                            gridspec_kw={'hspace' : 0.6})
    # fig.tight_layout()
    # Flatten axe structure to follow the sequence of observables.
    faxs = axs.flatten()

    # Make it iterable
    if nrows == 1 and ncols == 1:
        axs = [axs]
    # elif nrows == 1 or ncols == 1:
    #     axs = axs.flatten()

    # Some observables demand two subplots.
    nextaxis = 0

    if 'centroid' in observables:
        plot_double_observable(faxs, nextaxis, dataset, 'centroid',
                               first_item, last_item, droi=droi)
        observables.remove('centroid')
        nextaxis += 2

    if 'fwhm' in observables:
        plot_double_observable(faxs, nextaxis, dataset, 'fwhm',
                               first_item, last_item, droi=droi)
        observables.remove('fwhm')
        nextaxis += 2

    if 'intensity' in observables:
        for key, data in dataset.items():
            (motor, steps,
            xvals, yvals, sigmas) = observable_data(
                data, 'intensity', droi=droi
                )
            peaks, intens_by_mask, peak_by_fwhm = yvals
            dataset_plot(faxs[nextaxis], xvals, peaks, key,
                        "Peak", motor, first_item, last_item)
            dataset_plot(faxs[nextaxis+1], xvals, intens_by_mask, key,
                        "Masked intensity", motor, first_item, last_item)
            dataset_plot(faxs[nextaxis+2], xvals, peak_by_fwhm, key,
                        "Peak at FWHM", motor, first_item, last_item)
        observables.remove('intensity')
        nextaxis += 3

    # Loop over each observable and dataset to plot the
    # observable vs. step number.
    for idx, observable in enumerate(observables):
        for key, data in dataset.items():
            (motor, steps,
             xvals, yvals, sigmas) = observable_data(data,
                                                     observable,
                                                     droi=droi)
            for yval in yvals:
                dataset_plot(
                    faxs[nextaxis+idx], xvals, yval, key,
                    observable, motor, first_item, last_item
                    )

    plt.show()


# Old code for caustic analysis. To be checked.

def caustic_analysis(filename, filedir):
    """Perform caustic analysis on the given scan data."""
    file = '/'.join([filedir, filename])

    f = h5py.File(name=file, mode='r')

    positions = [f[scaname]['dvf_B1'].attrs['z_pos'] for scaname in f]

    caustic3d = []
    for scaname in f:
        img = np.array(f[scaname]['dvf_B1'])
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
