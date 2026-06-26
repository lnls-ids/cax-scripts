"""Set of methods to analyze scan data and extract beam properties.

This module provides classes to analyze CARCARÁ-X beamline data obtained
from scanning with its motors. All data is expected to be stored in HDF5
files.

The analysis operates at three hierarchical levels:

- **DataStep**:   A single scan step, containing raw image data, metadata,
  and computed beam properties for that step.
- **DataScan**:   A full scan (one pass), composed of multiple DataStep
  objects (e.g., all steps from one HDF5 file).
- **DataSet**:    Multiple scan passes of the same type, containing multiple
  DataScan objects (e.g., all HDF5 files from repeated scans).

Usage::

    from caxscripts.scananalysis import DataSet

    tx_passes = DataSet("/path/to/data", pattern=r"tx_scan[0-9]+")

    tx_01 = tx_passes.scans[0]
    tx_01.observables = ["fwhm", "centroid", "ry"]
    tx_01.plot_observables()
    tx_01.scan_animation(filename="tx_pass01.gif")
"""

import os
import re
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation
from   IPython.display import HTML
from   IPython.display import display as ipydisplay

from scipy.optimize import curve_fit

from . import utils
from caxscripts.image_statistics import Histogram2DAnalyzer

# Threshold for peak-to-average ratio acceptance of image.
THRESHOLD = 100

# Registry of multi-component observables.
# Each key can be used as an observable name; it expands to multiple
# subplot slots in plot_observables().
_COMPONENT_MAP = {
    'centroid':  ['centroid_x', 'centroid_y'],
    'fwhm':      ['fwhm_x',     'fwhm_y'],
    'intensity': ['intensity_peak', 'intensity_mask', 'intensity_fwhm_norm'],
}
_COMPONENT_FLAT = [v for vals in _COMPONENT_MAP.values() for v in vals]

# Recognised scan types.
_SCAN_TYPE_MIRROR  = 'mirror'
_SCAN_TYPE_SLIT    = 'slit'
_SCAN_TYPE_CAUSTIC = 'caustic'


# ===========================================================================
#  Helper utilities
# ===========================================================================

def _extract_metadata_value(val):
    """Return a scalar float from a metadata entry (array, list, or scalar)."""
    if isinstance(val, (list, np.ndarray)):
        return float(val[0])
    return float(val)

def _group_all_attrs(step_data):
    """
    Group all attributes of a step entry in the HDF5 file
    (identified as ``'attrs'``) into a single dictionary.
    """
    if 'attrs' in step_data:
        step_attrs = step_data['attrs']
    for key in step_data.keys():
        if 'attrs' in step_data[key]:
            extra_attrs = step_data[key]['attrs']
            step_attrs.update([
                (f"{key}.{k}", v) for k, v in extra_attrs.items()
            ])

    return step_attrs

def _find_slit_blade_device(metadata_list, scan_device):
    """Find which device's blade positions actually vary across steps.

    Scans metadata from all steps and returns the device prefix whose
    ``.top`` / ``.bottom`` / ``.left`` / ``.right`` keys are non-constant.
    Falls back to *scan_device* when no varying device is found.
    """
    _BLADE_KEYS = ('top', 'bottom', 'left', 'right')

    # Collect all device prefixes that have blade-position keys.
    candidates = set()
    for meta in metadata_list:
        for key in meta:
            parts = key.rsplit('.', 1)
            if len(parts) == 2 and parts[1] in _BLADE_KEYS:
                candidates.add(parts[0])

    if scan_device and scan_device not in candidates:
        candidates.add(scan_device)

    # Try scan_device first, then others, pick one that actually varies.
    ordered = ([scan_device] + sorted(candidates - {scan_device})
               if scan_device else sorted(candidates))
    for dev in ordered:
        cx_set = set()
        cy_set = set()
        for meta in metadata_list:
            vals = {}
            for side in _BLADE_KEYS:
                v = meta.get(f'{dev}.{side}')
                if v is not None:
                    vals[side] = _extract_metadata_value(v)
            if len(vals) == 4:
                cx_set.add(round((vals['left'] + vals['right']) / 2, 6))
                cy_set.add(round((vals['top'] + vals['bottom']) / 2, 6))
        if len(cx_set) > 1 or len(cy_set) > 1:
            return dev
    return scan_device


def _scan_variable_from_metadata(metadata, scan_type, scan_device, scan_motor,
                                  blade_device=None):
    """Extract the scanned-variable value for a single step.

    For standard scans reads ``{scan_device}.{scan_motor}`` from metadata.
    For slit scans computes the centre position from the four blade
    positions stored in the metadata.  Uses *blade_device* when given
    (so the caller can override a mis-matched ``scan_device`` attribute).
    """
    # if scan_type == _SCAN_TYPE_SLIT:
    #     dev    = blade_device or scan_device
    #     top    = _extract_metadata_value(metadata.get(f'{dev}.top',    0))
    #     bottom = _extract_metadata_value(metadata.get(f'{dev}.bottom', 0))
    #     left   = _extract_metadata_value(metadata.get(f'{dev}.left',   0))
    #     right  = _extract_metadata_value(metadata.get(f'{dev}.right',  0))
        
    #     # @Arnaldo, I'm not sure this calculation of slit center is correct.
    #     slit_center_x = (left + right) / 2
    #     slit_center_y = (top + bottom) / 2
    #     return (slit_center_x, slit_center_y)

    dev_motor = f'{scan_device}.{scan_motor}'
    val = metadata.get(dev_motor)
    if val is None:
        device_data = metadata.get(scan_device, {})
        if isinstance(device_data, dict):
            val = device_data.get(scan_motor)
    if val is None:
        return None
    return _extract_metadata_value(val)


def _build_beam_properties(analyzer, analysis_mode, exptime=1.0, droi=4):
    """Build a flat beam-properties dict from a Histogram2DAnalyzer.

    Returns
    -------
    dict or None
        None when the beam is not visible.
    """
    _MODE2HPRM = {'quick': 'qck', 'moments': 'mom', 'fit': 'fit', 'all': 'all'}
    hprm_key = _MODE2HPRM.get(analysis_mode, analysis_mode)

    if not analyzer.beam_visible:
        return None

    hprm = getattr(analyzer, f'hprm_{hprm_key}', None)
    if hprm is None:
        return None

    mu_x, mu_y = hprm['mux'], hprm['muy']
    f_x, f_y     = hprm['fwhmx'], hprm['fwhmy']
    s_x, s_y     = hprm['sigx'], hprm['sigy']

    props = {
        'centroid':   (mu_x, mu_y),
        'centroid_x': mu_x,
        'centroid_y': mu_y,
        'fwhm':       (f_x, f_y),
        'fwhm_x':     f_x,
        'fwhm_y':     f_y,
        'sigma':      (s_x, s_y),
        'sigma_x':    s_x,
        'sigma_y':    s_y,
        'beam_visible': analyzer.beam_visible,
    }

    for key in ('cov', 'sig_major', 'sig_minor', 'theta', 'evecs',
                'xcenters', 'ycenters'):
        if key in hprm:
            props[key] = hprm[key]

    # Intensity calculations (three methods).
    img = analyzer.img
    cx, cy = int(mu_x), int(mu_y)
    peak = np.mean(img[cx - droi:cx + droi + 1,
                       cy - droi:cy + droi + 1])
    peak /= exptime
    peak_fwhm_norm = peak / (f_x * f_y) if f_x * f_y != 0 else 0

    mask = img > (peak * exptime / 2)
    area_mask = np.sum(mask)
    img_masked = np.where(mask, img, 0)
    area_img_masked = np.sum(img_masked)
    intensity_by_mask = (area_img_masked / (area_mask * exptime)
                         if area_mask != 0 else 0)

    props['intensity'] = {
        'peak':       peak,
        'mask':       intensity_by_mask,
        'fwhm_norm':  peak_fwhm_norm,
    }
    props['intensity_peak']      = peak
    props['intensity_mask']      = intensity_by_mask
    props['intensity_fwhm_norm'] = peak_fwhm_norm

    return props


# ========================================================================
#  DataStep
# ========================================================================

class DataStep:
    """A single step in a scan.

    Stores the raw image, all metadata (motor positions, device settings),
    and computes beam properties **eagerly** on initialisation.  The
    underlying :class:`~caxscripts.image_statistics.Histogram2DAnalyzer`
    is retained as the ``.analyzer`` attribute for advanced use (e.g.
    re-analysis with a different mode, plotting with ellipses).

    Parameters
    ----------
    step_index : int
        Zero-based index within its scan.
    metadata : dict
        All group attributes from the HDF5 file for this step.
    image : numpy.ndarray or None
        Raw 2D image data from the DVF detector (primary).
    image_secondary : numpy.ndarray or None
        Raw 2D image from a second DVF detector (e.g. for slit animation).
    scan_type : str
        ``'mirror'``, ``'slit'``, or ``'caustic'``.
    scan_device : str or None
        Device being scanned (e.g. ``'mirror'``, ``'slit_A1'``).
    scan_motor : str or None
        Motor being scanned (e.g. ``'tx'``, ``'z_pos'``).
    analysis_mode : str
        ``'quick'``, ``'moments'``, or ``'fit'``.
    droi : int
        Half-size of the region of interest for beam visibility.
    exptime : float
        Exposure time in seconds (used for intensity normalisation).
    """

    def __init__(self, step_index, metadata, image=None, image_slit=None,
                 scan_type=None, scan_device=None, scan_motor=None,
                 analysis_mode='quick', droi=4, exptime=1.0,
                 blade_device=None):
        self.step_index          = step_index
        self.metadata            = metadata
        self.image_slit          = image_slit
        self.scan_variable_value = None
        self.beam_properties     = None
        self.analyzer            = None

        self.scan_variable_value = _scan_variable_from_metadata(
            metadata, scan_type, scan_device, scan_motor,
            blade_device=blade_device,
        )

        if image is not None:
            img_arr = np.asarray(image.T, dtype=float)
            nx, ny = img_arr.shape
            xedges = np.arange(nx + 1)
            yedges = np.arange(ny + 1)
            self.analyzer = Histogram2DAnalyzer(
                img_arr, xedges, yedges, droi=droi,
            )
            if self.analyzer.beam_visible:
                self.analyzer.analyze(analysis_mode)
                self.beam_properties = _build_beam_properties(
                    self.analyzer, analysis_mode,
                    exptime=exptime, droi=droi,
                )

    # ------------------------------------------------------------------
    #  Image access — backed by analyzer to avoid duplicating pixel data
    # ------------------------------------------------------------------

    @property
    def image(self):
        """Primary beam image (float64).  Backed by ``self.analyzer.img``."""
        if self.analyzer is not None:
            return self.analyzer.img
        return None

    # ------------------------------------------------------------------
    #  Image plotting
    # ------------------------------------------------------------------

    def plot_image(self, analysis_mode=None, **kwargs):
        """Plot the beam image with fit ellipses.

        Delegates to :meth:`Histogram2DAnalyzer.plot`.

        Parameters
        ----------
        analysis_mode : str or None
            Which HPRM dict to use (``'quick'``, ``'moments'``, ``'fit'``).
            Defaults to the mode used during construction.
        **kwargs
            Forwarded to ``Histogram2DAnalyzer.plot()``.

        Returns
        -------
        fig, ax
        """
        if self.analyzer is None:
            raise RuntimeError("No image data available to plot.")
        mode = analysis_mode
        hprm = None
        if mode is not None:
            hprm = getattr(self.analyzer, f'hprm_{mode}', None)
        return self.analyzer.plot(hprm=hprm, **kwargs)

    def __repr__(self):
        visible = (self.beam_properties is not None
                   and self.beam_properties.get('beam_visible'))
        val = self.scan_variable_value
        if val is not None:
            if isinstance(val, tuple):
                return (f"<DataStep #{self.step_index}: "
                        f"centre=({val[0]:.3g},{val[1]:.3g}), "
                        f"{'visible' if visible else 'no beam'}>")
            return (f"<DataStep #{self.step_index}: "
                    f"var={val:.4g}, "
                    f"{'visible' if visible else 'no beam'}>")
        return f"<DataStep #{self.step_index}: {'visible' if visible else 'no beam'}>"


# ===========================================================================
#  DataScan
# ===========================================================================

class DataScan:
    """One full scan pass, composed of individual :class:`DataStep` objects.

    Parameters
    ----------
    scan_index : int
        Index of this scan within a set.
    scan_type : str
        ``'mirror'``, ``'slit'``, ``'caustic'``.
    scan_variable : str
        Name of the scanned variable (e.g. ``'mirror.tx'``).
    scan_dict: dict
        Dictionary containing all scan metadata and steps.
    scan_device : str or None
        Device being scanned.
    scan_motor : str or None
        Motor being scanned.
    """

    def __init__(self, scan_index, scan_type, scan_variable,
                 scan_dict, scan_device, scan_motor, scan_name,
                 analysis_mode, _droi):
        self.scan_index    = scan_index
        self.scan_type     = scan_type
        self.scan_variable = scan_variable
        self.scan_device   = scan_device
        self.scan_motor    = scan_motor
        self.scan_name     = scan_name
        self.analysis_mode = analysis_mode
        self._droi         = _droi 
        self.observables   = []
        self.step_range    = None       # (start, end) tuple for slicing

        self._load(scan_dict)
    # ------------------------------------------------------------------
    #  Discovery
    # ------------------------------------------------------------------


    def _load(self, scan_dict):
        step_keys = sorted(scan_dict.keys())
        if not step_keys:
            raise ValueError("HDF5 file contains no scan groups.")

        self.steps = []
        for sk in step_keys:
            step_data  = scan_dict[sk]
            step_attrs = _group_all_attrs(step_data)

            step_idx = int(sk.split('-')[-1])

            image_b1 = step_data.get('dvf_B1', {}).get('data')
            image_a1 = step_data.get('dvf_A1', {}).get('data')
            exptime = 1.0
            if image_b1 is not None:
                exptime = step_data['dvf_B1']['attrs'].get('expo_time', 1.0)
            elif image_a1 is not None:
                exptime = step_data['dvf_A1']['attrs'].get('expo_time', 1.0)

            step = DataStep(
                step_index=step_idx,
                metadata=step_attrs,
                image=image_b1,               # primary: DVF B1
                image_slit=image_a1,     # secondary: DVF A1
                scan_type=self.scan_name,
                scan_device=self.scan_device,
                scan_motor=self.scan_motor,
                analysis_mode=self.analysis_mode,
                droi=self._droi,
                exptime=exptime,
                # blade_device=blade_device,
            )
            self.steps.append(step)
    
    @property
    def available_observables(self):
        """List all plottable observable names.

        Includes multi-component names (``'centroid'``, ``'fwhm'``,
        ``'intensity'``), their flat components, and every scalar metadata
        key present in the first step.
        """
        if not self.steps:
            return []
        meta = self.steps[0].metadata
        meta_keys = []
        for k, v in meta.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                meta_keys.append(k)
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                meta_keys.append(k)
            elif isinstance(v, np.ndarray) and v.ndim == 1:# and v.size == 1:
                meta_keys.append(k)

        return list(_COMPONENT_MAP.keys()) + _COMPONENT_FLAT + meta_keys

    def describe(self):
        """Print an overview of this scan and its available observables."""
        name = self.scan_variable or '(unknown)'
        print(f"Scan #{self.scan_index}  [{self.scan_type}]")
        print(f"  variable : {name}")
        print(f"  steps    : {len(self.steps)}")
        obs = self.available_observables
        print(f"  observables ({len(obs)}):")
        for o in obs: print("\t"+o, end=";\n")

    # ------------------------------------------------------------------
    #  Internal data helpers
    # ------------------------------------------------------------------

    def _steps_in_range(self):
        if self.step_range is None:
            return self.steps
        start, end = self.step_range
        if end == -1:
            return self.steps[start:]
        return self.steps[start:end]

    def _get_observable_value(self, step, observable):
        """Return the scalar value of *observable* for a single step."""
        # 1. beam_properties.
        if step.beam_properties and observable in step.beam_properties:
            val = step.beam_properties[observable]
            if isinstance(val, (list, np.ndarray, tuple)):
                return float(val[0]) if len(val) else float('nan')
            return float(val)

        # 2. metadata.
        val = step.metadata.get(observable)
        if val is not None:
            return _extract_metadata_value(val)

        return float('nan')

    def resolve_observable(self, observable):
        """Return ``(x_values, y_values)`` arrays for an observable.

        Multi-component observables (``'centroid'``, ``'fwhm'``,
        ``'intensity'``) return *y_values* as a *list* of arrays (one per
        component).  Scalar observables return a single 1-D array.

        Parameters
        ----------
        observable : str

        Returns
        -------
        xvals : np.ndarray
            Scanned-variable value for each step.
        yvals : np.ndarray or list of np.ndarray
        """
        if observable in _COMPONENT_MAP:
            components = _COMPONENT_MAP[observable]
            results = [self.resolve_observable(c) for c in components]
            # Returns xvals (np.ndarray) and yvals (list[np.ndarray])
            return results[0][0], [r[1] for r in results]

        steps = self._steps_in_range()
        xvals = np.array([
            (step.scan_variable_value if not isinstance(step.scan_variable_value, tuple)
             else step.scan_variable_value[0])
            if step.scan_variable_value is not None else float('nan')
            for step in steps
        ])
        yvals = np.array([self._get_observable_value(step, observable)
                          for step in steps])
        return xvals, yvals

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------

    def plot_observables(self, observables=None, first_item=0, last_item=None,
                         droi=8):
        """Plot selected observables versus the scanned variable.

        Multi-component observables (``'centroid'``, ``'fwhm'``,
        ``'intensity'``) expand into separate subplots per component.
        Slit scans use a special heatmap-style plot.

        Parameters
        ----------
        observables : list of str or None
            If None, uses ``self.observables``.
        first_item : int
            First step index to include, **relative to the** ``step_range``
            window.  E.g. ``step_range=(2, 10)`` + ``first_item=1`` means
            steps 3–9 are plotted.
        last_item : int or None
            Last step index (exclusive).  None = all remaining steps in
            the ``step_range`` window.
        droi : int
            ROI half-size (used by some analysis steps).

        Returns
        -------
        fig, axs
        """
        if self.scan_type == _SCAN_TYPE_SLIT:
            raise NotImplementedError()
            # return self._plot_slit(observables, first_item, last_item)

        return self._plot_default(observables, first_item, last_item)

    def _plot_default(self, observables, first_item=0, last_item=None):
        """Standard line-plot layout for mirror / general scans."""
        observable_names = observables if observables is not None \
                           else self.observables
        if not observable_names:
            print("No observables set. Set scan.observables or pass them.")
            return None, None

        all_observables = []
        for obs in observable_names:
            all_observables.extend(_COMPONENT_MAP.get(obs, [obs]))

        # Setting up plot parameters
        n_plots = len(all_observables)
        nrows = max((n_plots + 1) // 2, 1)
        ncols = 2 if n_plots > 1 else 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(10 * ncols, 6 * nrows))
        if n_plots == 1:
            axs = np.array([axs])
        # Flattening the axes simplifies indexing logic
        ax_flat = axs.flatten()

        # Scanned variable label
        var_label = (self.scan_variable.split('.')[-1]
                     if self.scan_variable and '.' in self.scan_variable
                     else self.scan_variable or 'step')

        for i, obs_name in enumerate(all_observables):
            ax = ax_flat[i]
            xvals, yvals = self.resolve_observable(obs_name)
            sl = slice(first_item, last_item)
            x_sliced = xvals[sl]

            # Multi component observables
            if isinstance(yvals, list):
                for yvals_i in yvals:
                    y_sliced = yvals_i[sl]
                    ax.plot(x_sliced, y_sliced, marker='o', label=obs_name)
            else:
                y_sliced = yvals[sl]
                ax.plot(x_sliced, y_sliced, marker='o', label=obs_name)

            ax.set_xlabel(var_label)
            ax.set_ylabel(obs_name)
            ax.set_title(obs_name.replace('_', ' ').capitalize())
            ax.legend()
            ax.grid(True)

        # Hide remaining unused axes
        for j in range(n_plots, len(ax_flat)):
            ax_flat[j].set_visible(False)

        plt.tight_layout()
        return fig, axs

    #######
    # A Wasp's nest -- Not usable yet, to be replaced by a more robust method
    ###### 
    # def _plot_slit(self, observables, first_item=0, last_item=None):
    #     """Special plot for slit scans: heatmap of observables.

    #     The grid positions are the slit centre (cx, cy) computed from
    #     the four blade positions.  If the centres form a regular grid
    #     (i.e. every unique cx is paired with every unique cy), a
    #     :func:`pcolormesh` heatmap is drawn.  Otherwise a scatter plot
    #     is used as fallback.

    #     When multiple steps share the same (cx, cy) their observable
    #     values are averaged (``np.nanmean``).  Steps where the beam is
    #     not visible contribute NaN and appear transparent.
    #     """
    #     obs = observables if observables is not None else self.observables
    #     if not obs:
    #         print("No observables set.")
    #         return None, None

    #     steps = self._steps_in_range()[first_item:last_item]

    #     # Collect centre positions.
    #     cx_vals = []
    #     cy_vals = []
    #     for s in steps:
    #         if isinstance(s.scan_variable_value, tuple):
    #             cx_vals.append(s.scan_variable_value[0])
    #             cy_vals.append(s.scan_variable_value[1])
    #         else:
    #             cx_vals.append(s.scan_variable_value or 0)
    #             cy_vals.append(0)

    #     cx_arr = np.array(cx_vals)
    #     cy_arr = np.array(cy_vals)

    #     expanded = []
    #     for o in obs:
    #         expanded.extend(_COMPONENT_MAP.get(o, [o]))

    #     # --- Build grid ---
    #     # Try exact positions first, then rounding to handle jitter,
    #     # then step_col/step_row metadata.

    #     def _try_grid_from_positions(cx, cy):
    #         """Return (cx_u, cy_u, cx_idx, cy_idx) or None."""
    #         cx_u = np.unique(cx)
    #         cy_u = np.unique(cy)
    #         unique_pairs = set(zip(
    #             tuple(round(float(cx[k]), 6) for k in range(len(cx))),
    #             tuple(round(float(cy[k]), 6) for k in range(len(cy))),
    #         ))
    #         if cx_u.size * cy_u.size == len(unique_pairs):
    #             cx_map = {v: j for j, v in enumerate(cx_u)}
    #             cy_map = {v: i for i, v in enumerate(cy_u)}
    #             cx_idx = np.array([cx_map[v] for v in cx])
    #             cy_idx = np.array([cy_map[v] for v in cy])
    #             return cx_u, cy_u, cx_idx, cy_idx
    #         return None

    #     def _try_grid_from_rounded(cx, cy):
    #         """Try rounding to fewer decimals to form a regular grid."""
    #         for ndec in range(5, 1, -1):
    #             cx_r = np.round(cx, ndec)
    #             cy_r = np.round(cy, ndec)
    #             cx_u = np.unique(cx_r)
    #             cy_u = np.unique(cy_r)
    #             unique_pairs = set(zip(
    #                 tuple(round(float(cx_r[k]), ndec) for k in range(len(cx))),
    #                 tuple(round(float(cy_r[k]), ndec) for k in range(len(cy))),
    #             ))
    #             if cx_u.size * cy_u.size == len(unique_pairs):
    #                 cx_map = {v: j for j, v in enumerate(cx_u)}
    #                 cy_map = {v: i for i, v in enumerate(cy_u)}
    #                 cx_idx = np.array([cx_map[v] for v in cx_r])
    #                 cy_idx = np.array([cy_map[v] for v in cy_r])
    #                 return cx_u, cy_u, cx_idx, cy_idx
    #         return None

    #     def _try_grid_from_step_meta(steps, cx, cy):
    #         """Use step_col / step_row as grid indices."""
    #         col_vals = []
    #         row_vals = []
    #         mask = []
    #         for k, s in enumerate(steps):
    #             m = s.metadata
    #             cv = m.get('step_col', 'N/A')
    #             rv = m.get('step_row', 'N/A')
    #             if cv == 'N/A' or rv == 'N/A':
    #                 mask.append(False)
    #                 continue
    #             try:
    #                 col_vals.append(int(cv))
    #                 row_vals.append(int(rv))
    #                 mask.append(True)
    #             except (ValueError, TypeError):
    #                 mask.append(False)
    #         if sum(mask) < 2:
    #             return None
    #         col_arr = np.array(col_vals)
    #         row_arr = np.array(row_vals)
    #         # Build full grid and check occupancy.
    #         full = set()
    #         for c in np.unique(col_arr):
    #             for r in np.unique(row_arr):
    #                 full.add((c, r))
    #         present = set(zip(col_arr, row_arr))
    #         if len(present) < len(full):
    #             return None
    #         col_u = np.unique(col_arr)
    #         row_u = np.unique(row_arr)
    #         # Filter original data to grid steps only
    #         mask_arr = np.array(mask)
    #         cx_f = cx[mask_arr]
    #         cy_f = cy[mask_arr]
    #         return (col_u, row_u, col_arr, row_arr,
    #                 cx_f, cy_f, mask_arr)

    #     grid = _try_grid_from_positions(cx_arr, cy_arr)
    #     grid_mode = 'positions'

    #     if grid is None:
    #         grid = _try_grid_from_rounded(cx_arr, cy_arr)
    #         grid_mode = 'rounded'

    #     if grid is None:
    #         grid = _try_grid_from_step_meta(steps, cx_arr, cy_arr)
    #         grid_mode = 'step_meta'

    #     nplots = len(expanded)
    #     fig, axs = plt.subplots(1, nplots, figsize=(6 * nplots, 5),
    #                             squeeze=False)

    #     use_pcolormesh = grid is not None

    #     for i, name in enumerate(expanded):
    #         ax = axs[0, i]
    #         yv = np.array(
    #             [self._get_observable_value(s, name) for s in steps])

    #         if use_pcolormesh:
    #             if grid_mode == 'step_meta':
    #                 col_u, row_u, col_arr, row_arr, cx_f, cy_f, mask_arr = grid
    #                 yv_f = yv[mask_arr]  # filter to grid steps only
    #                 ny, nx = row_u.size, col_u.size
    #                 Z_sum = np.zeros((ny, nx))
    #                 Z_cnt = np.zeros((ny, nx))
    #                 cx_grid = np.full((ny, nx), np.nan)
    #                 cy_grid = np.full((ny, nx), np.nan)
    #                 col_idx = {v: j for j, v in enumerate(col_u)}
    #                 row_idx = {v: i for i, v in enumerate(row_u)}
    #                 for k in range(len(col_arr)):
    #                     v = yv_f[k]
    #                     ci = row_idx[row_arr[k]]
    #                     cj = col_idx[col_arr[k]]
    #                     cx_grid[ci, cj] = cx_f[k]
    #                     cy_grid[ci, cj] = cy_f[k]
    #                     if not np.isnan(v):
    #                         Z_sum[ci, cj] += v
    #                         Z_cnt[ci, cj] += 1
    #                 Z = np.where(Z_cnt > 0, Z_sum / Z_cnt, np.nan)
    #                 CX = np.nanmean(cx_grid, axis=0)
    #                 CY = np.nanmean(cy_grid, axis=1)
    #                 CX, CY = np.meshgrid(CX, CY)
    #             else:
    #                 cx_u, cy_u, cx_idx, cy_idx = grid
    #                 ny, nx = cy_u.size, cx_u.size
    #                 Z_sum = np.zeros((ny, nx))
    #                 Z_cnt = np.zeros((ny, nx))
    #                 for k in range(len(steps)):
    #                     v = yv[k]
    #                     if not np.isnan(v):
    #                         Z_sum[cy_idx[k], cx_idx[k]] += v
    #                         Z_cnt[cy_idx[k], cx_idx[k]] += 1
    #                 Z = np.where(Z_cnt > 0, Z_sum / Z_cnt, np.nan)
    #                 CX, CY = np.meshgrid(cx_u, cy_u)

    #             mesh = ax.pcolormesh(CX, CY, Z, shading='auto', cmap='viridis')
    #             plt.colorbar(mesh, ax=ax, label=name)
    #         else:
    #             sc = ax.scatter(cx_arr, cy_arr, c=yv, cmap='viridis',
    #                             s=80, edgecolor='k')
    #             plt.colorbar(sc, ax=ax, label=name)

    #         ax.set_xlabel('Slit centre X')
    #         ax.set_ylabel('Slit centre Y')
    #         ax.set_title(name.replace('_', ' ').capitalize())
    #         ax.grid(True)

    #     plt.tight_layout()
    #     return fig, axs

    # ------------------------------------------------------------------
    #  Statistics
    # ------------------------------------------------------------------

    def mean_value(self, observable):
        """Mean of an observable across steps.

        Returns a dict for multi-component observables, else a float.
        """
        _, yvals = self.resolve_observable(observable)
        if isinstance(yvals, list):
            return {f'{observable}_{i}': float(np.nanmean(yvals_i))
                    for i, yvals_i in enumerate(yvals)}
        return float(np.nanmean(yvals))

    def std_deviation(self, observable):
        """Standard deviation of an observable across steps."""
        _, yvals = self.resolve_observable(observable)
        if isinstance(yvals, list):
            return {f'{observable}_{i}': float(np.nanstd(yvals_i))
                    for i, yvals_i in enumerate(yvals)}
        return float(np.nanstd(yvals))

    # ------------------------------------------------------------------
    #  Animation
    # ------------------------------------------------------------------

    def scan_animation(self, observables=None, filename=None, fps=2,
                       save_fmt='gif'):
        """Animate beam images with observable trace markers.

        Slit scans use a dual-image layout (DVF A1 + DVF B1).
        Other scans use a single-image layout.

        Parameters
        ----------
        observables : list of str or None
            If None, uses ``self.observables[:1]``.
        filename : str or None
            If provided, save to file; otherwise display inline.
        fps : int
            Frames per second.
        save_fmt : str
            ``'gif'`` or ``'mp4'``.
        """
        if self.scan_type == _SCAN_TYPE_SLIT:
            # return self._animate_slit(observables, filename, fps, save_fmt)
            raise NotImplementedError()
        return self._animate_default(observables, filename, fps, save_fmt)

    def _animate_default(self, observable, filename, fps, save_fmt):
        """Standard single-image animation."""
        if observable is not None:
            obs_name = observable
        else:
            obs_name = self.observables[:1]
            print(f"Warning: no observable set, using: {obs_name}")
        if not obs_name:
            print("No observables set.")
            return

        steps = self._steps_in_range()
        xvals, yvals_list = self.resolve_observable(obs_name)

        images   = [step.image if step.image is not None
                    else np.zeros((10, 10)) for step in steps]
        n_frames = len(images)

        # Determine the number of subplots (depending on how many
        # observable components are present)
        n_trace   = len(yvals_list) if isinstance(yvals_list, list) else 1
        fig, axes = plt.subplots(1 + n_trace, 1,
                                 figsize=(10, 4 + 3 * n_trace))
        if 1 + n_trace == 1:
            axes = [axes]
        ax_img = axes[0]

        # Image plotting
        im = ax_img.imshow(images[0].T, cmap='viridis', animated=True)
        plt.colorbar(im, ax=ax_img, label='Intensity')
        ax_img.set_xlabel('Pixel X')
        ax_img.set_ylabel('Pixel Y')
        img_title = ax_img.set_title('Step 0')

        var_label = (self.scan_variable.split('.')[-1]
                     if self.scan_variable and '.' in self.scan_variable
                     else self.scan_variable or 'step')
        trace_markers = []

        # Separate axis to plot observable evolution along scan
        if isinstance(yvals_list, list):
            for i, (yvals_i, ax_tr) in enumerate(zip(yvals_list, axes[1:])):
                ax_tr.plot(xvals, yvals_i, marker='o', color='steelblue',
                           label=f'{obs_name} {i}')
                mk, = ax_tr.plot([], [], 'ro', markersize=10, label='Current')
                trace_markers.append(mk)
                ax_tr.set_xlabel(var_label)
                ax_tr.set_ylabel(f'{obs_name} {i}')
                ax_tr.legend()
                ax_tr.grid(True)
        else:
            ax_tr = axes[1]
            ax_tr.plot(xvals, yvals_list, marker='o', color='steelblue',
                       label=obs_name)
            trace_markers.append(
                ax_tr.plot([], [], 'ro', markersize=10, label='Current')[0]
            )
            ax_tr.set_xlabel(var_label)
            ax_tr.set_ylabel(obs_name)
            ax_tr.legend()
            ax_tr.grid(True)

        def update(frame):
            im.set_data(images[frame].T)
            im.set_clim(images[frame].min(), images[frame].max())
            img_title.set_text(f'Step {frame}')
            for j, mk in enumerate(trace_markers):
                if isinstance(yvals_list, list):
                    mk.set_data([xvals[frame]], [yvals_list[j][frame]])
                else:
                    mk.set_data([xvals[frame]], [yvals_list[frame]])
            return [im, img_title] + trace_markers

        anim = FuncAnimation(fig, update, frames=n_frames,
                             interval=1000 // fps, blit=True)
        plt.tight_layout()

        if filename:
            if save_fmt == 'gif':
                anim.save(filename, writer='pillow', fps=fps, dpi=300)
            elif save_fmt == 'mp4':
                anim.save(filename, writer='ffmpeg', fps=fps, dpi=300)
            plt.close()
        else:
            plt.close()
            ipydisplay(HTML(anim.to_jshtml()))

        return anim

    def _animate_slit(self, observables, filename, fps, save_fmt):
        """Dual-image animation for slit scans (DVF A1 + DVF B1).

        Uses step index as the x-axis on the trace plots (since slit scans
        vary two axes simultaneously and have no single 1-D motor variable).
        """
        obs = observables if observables is not None else self.observables[:1]
        if not obs:
            print("No observables set.")
            return

        obs_name = obs[0]
        steps = self._steps_in_range()
        xv = np.array([s.step_index for s in steps])
        _, yv_list = self.resolve_observable(obs_name)

        images_a1 = [s.image_secondary if s.image_secondary is not None
                     else np.zeros((10, 10)) for s in steps]
        images_b1 = [s.image if s.image is not None
                     else np.zeros((10, 10)) for s in steps]
        n_frames = len(images_a1)

        n_trace = len(yv_list) if isinstance(yv_list, list) else 1
        fig = plt.figure(figsize=(14, 4 + 3 * n_trace))
        gs = fig.add_gridspec(1 + n_trace, 2, hspace=0.3, wspace=0.3)
        ax_a1 = fig.add_subplot(gs[0, 0])
        ax_b1 = fig.add_subplot(gs[0, 1])

        im_a1 = ax_a1.imshow(images_a1[0], cmap='viridis', animated=True)
        plt.colorbar(im_a1, ax=ax_a1, label='Intensity')
        ax_a1.set_xlabel('Pixel X')
        ax_a1.set_ylabel('Pixel Y')
        ax_a1.set_title('DVF A1')

        im_b1 = ax_b1.imshow(images_b1[0], cmap='viridis', animated=True)
        plt.colorbar(im_b1, ax=ax_b1, label='Intensity')
        ax_b1.set_xlabel('Pixel X')
        ax_b1.set_ylabel('Pixel Y')
        ax_b1.set_title('DVF B1')

        step_label = fig.suptitle('Step 0', y=0.95)

        trace_markers = []

        if isinstance(yv_list, list):
            for i, (yvi, idx) in enumerate(zip(yv_list, range(n_trace))):
                ax_tr = fig.add_subplot(gs[1 + idx, :])
                ax_tr.plot(xv, yvi, marker='o', color='steelblue',
                           label=f'{obs_name} {i}')
                mk, = ax_tr.plot([], [], 'ro', markersize=10, label='Current')
                trace_markers.append(mk)
                ax_tr.set_xlabel('Step index')
                ax_tr.set_ylabel(f'{obs_name} {i}')
                ax_tr.legend()
                ax_tr.grid(True)
        else:
            ax_tr = fig.add_subplot(gs[1, :])
            ax_tr.plot(xv, yv_list, marker='o', color='steelblue',
                       label=obs_name)
            trace_markers.append(
                ax_tr.plot([], [], 'ro', markersize=10, label='Current')[0]
            )
            ax_tr.set_xlabel('Step index')
            ax_tr.set_ylabel(obs_name)
            ax_tr.legend()
            ax_tr.grid(True)

        def update(frame):
            im_a1.set_data(images_a1[frame])
            im_a1.set_clim(images_a1[frame].min(), images_a1[frame].max())
            im_b1.set_data(images_b1[frame])
            im_b1.set_clim(images_b1[frame].min(), images_b1[frame].max())
            step_label.set_text(f'Step {frame}')
            for j, mk in enumerate(trace_markers):
                if isinstance(yv_list, list):
                    mk.set_data([xv[frame]], [yv_list[j][frame]])
                else:
                    mk.set_data([xv[frame]], [yv_list[frame]])
            return [im_a1, im_b1, step_label] + trace_markers

        anim = FuncAnimation(fig, update, frames=n_frames,
                             interval=1000 // fps, blit=True)

        if filename:
            if save_fmt == 'gif':
                anim.save(filename, writer='pillow', fps=fps, dpi=300)
            elif save_fmt == 'mp4':
                anim.save(filename, writer='ffmpeg', fps=fps, dpi=300)
            plt.close()
        else:
            plt.close()
            ipydisplay(HTML(anim.to_jshtml()))

        return anim

    def __repr__(self):
        return (f"<DataScan #{self.scan_index}: {self.scan_type}, "
                f"{len(self.steps)} steps>")


# ===========================================================================
#  DataSet
# ===========================================================================

class DataSet:
    """A collection of repeated scans of the same type.

    Loads HDF5 files matching a pattern, parses them into :class:`DataScan`
    and :class:`DataStep` objects, and validates that all scans share the
    same ``scan_type``.

    Parameters
    ----------
    workdir : str
        Directory containing the HDF5 files.
    pattern : str
        Regex pattern to match filenames.
    analysis_mode : str
        ``'quick'``, ``'moments'``, or ``'fit'``.
    droi : int
        ROI half-size for beam detection.
    """

    def __init__(self, workdir, pattern, analysis_mode='quick', droi=4):
        self.scans         = []
        self.scan_type     = None
        self.scan_variable = None
        self.scan_device   = None
        self.scan_motor    = None

        # Global slicing for plot / statistics methods.
        self.scan_range = None          # (start, end) tuple
        self.step_range = None          # (start, end) tuple

        self.analysis_mode = analysis_mode
        self._droi = droi

        self._load(workdir, pattern)

    # ------------------------------------------------------------------
    #  File I/O
    # ------------------------------------------------------------------

    @staticmethod
    def files_in_directory(wdir, pattern):
        """List files in *wdir* matching regex *pattern*."""
        raw = os.listdir(wdir)
        return sorted(os.path.join(wdir, f)
                      for f in raw if re.match(pattern, f))

    @staticmethod
    def h5_to_dict(filename):
        """Read an HDF5 file into a nested dict.

        Structure::

            data[group_name]['attrs']  -> group attributes
            data[group_name][dataset]  -> {'data': ndarray, 'attrs': ...}
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

    # ------------------------------------------------------------------
    #  Internal loading
    # ------------------------------------------------------------------

    def _load(self, workdir, pattern):
        """Load all matching HDF5 files and parse into DataScan objects."""
        files = self.files_in_directory(workdir, pattern)
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {workdir}"
            )

        for i, fpath in enumerate(files):
            raw = self.h5_to_dict(fpath)
            sd = self._parse_scan(raw, scan_index=i)
            self.scans.append(sd)

            if i == 0:
                self.scan_type = sd.scan_type
                self.scan_variable = sd.scan_variable
                self.scan_device = sd.scan_device
                self.scan_motor = sd.scan_motor
            elif sd.scan_type != self.scan_type:
                raise ValueError(
                    f"Scan #{i} has type '{sd.scan_type}', "
                    f"expected '{self.scan_type}'. "
                    "DataSet requires all scans to be of the same type."
                )

    def _parse_scan(self, scan_dict, scan_index):
        """Convert a raw HDF5 dict into a DataScan with DataStep children."""
        step_keys = sorted(scan_dict.keys())
        if not step_keys:
            raise ValueError("HDF5 file contains no scan groups.")

        first = scan_dict[step_keys[0]]
        attrs = first['attrs']

        scan_name = attrs.get('scan_name', 'mirror')
        if scan_name == _SCAN_TYPE_SLIT:
            scan_device   = attrs.get('scan_device', 'slit_A1')
            scan_motor    = None
            scan_variable = f'{scan_device}.centre'
        else:
            scan_device   = attrs.get('scan_device', 'mirror')
            scan_motor    = attrs.get('scan_motor', 'tx')
            scan_variable = f'{scan_device}.{scan_motor}'

        return DataScan(
            scan_index=scan_index,
            scan_type=scan_name,
            scan_variable=scan_variable,
            scan_dict=scan_dict,
            scan_device=scan_device,
            scan_motor=scan_motor,
            scan_name=scan_name,
            analysis_mode=self.analysis_mode,
            _droi=self._droi,
        )

    # ------------------------------------------------------------------
    #  Slicing helpers
    # ------------------------------------------------------------------

    def _scans_in_range(self):
        if self.scan_range is None:
            return self.scans
        start, end = self.scan_range
        if end == -1:
            return self.scans[start:]
        return self.scans[start:end]

    # ------------------------------------------------------------------
    #  Multi-scan plotting
    # ------------------------------------------------------------------

    def plot_all(self, observables=None):
        """Call ``plot_observables`` on each scan individually."""
        for scan in self._scans_in_range():
            scan.step_range = self.step_range
            scan.plot_observables(observables=observables)

    def plot_superimposed(self, observable, first_item=0, last_item=None):
        """Overlay an observable trace from all scans on a single axes.

        Parameters
        ----------
        observable : str
        first_item : int
        last_item : int or None

        Returns
        -------
        fig, ax
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        scans   = self._scans_in_range()

        for scan in scans:
            scan.step_range = self.step_range
            xv, yv = scan.resolve_observable(observable)
            sl = slice(first_item, last_item)
            label = f'Scan #{scan.scan_index}'
            if isinstance(yv, list):
                ax.plot(xv[sl], yv[0][sl], marker='o', label=label)
            else:
                ax.plot(xv[sl], yv[sl], marker='o', label=label)

        var_label = (self.scan_variable.split('.')[-1]
                     if self.scan_variable and '.' in self.scan_variable
                     else self.scan_variable or 'step')
        ax.set_xlabel(var_label)
        ax.set_ylabel(observable)
        ax.set_title(f'{observable} — {len(scans)} scans superimposed')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    #  Statistics
    # ------------------------------------------------------------------

    def statistics(self, observable):
        """Compute mean, median, and std of an observable across scans.

        Parameters
        ----------
        observable : str

        Returns
        -------
        dict
            Keys ``'xval'``, ``'mean'``, ``'median'``, ``'std_dev'``.
        """
        scans = self._scans_in_range()
        y_all = []
        xvals = None

        for scan in scans:
            scan.step_range = self.step_range
            xv, yv = scan.resolve_observable(observable)
            if xvals is None:
                xvals = xv
            y_all.append(yv[0] if isinstance(yv, list) else yv)

        y_all = np.array(y_all, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {
                'xval':    xvals,
                'mean':    np.nanmean(y_all, axis=0),
                'median':  np.nanmedian(y_all, axis=0),
                'std_dev': np.nanstd(y_all, axis=0),
            }

    def correlation_matrix(self, observable):
        """Pairwise Pearson correlation of *observable* between scans.

        Returns
        -------
        np.ndarray
            N x N correlation matrix.
        """
        scans = self._scans_in_range()
        n = len(scans)
        mat = np.eye(n)
        traces = []
        for scan in scans:
            scan.step_range = self.step_range
            _, yv = scan.resolve_observable(observable)
            traces.append(yv[0] if isinstance(yv, list) else yv)

        for i in range(n):
            for j in range(i + 1, n):
                c = np.corrcoef(traces[i], traces[j])[0, 1]
                mat[i, j] = mat[j, i] = c
        return mat

    def centroid_delta_plot(self, motor, step_start=0, step_end=-1):
        """Motor change vs centroid-X drift across scans.

        Parameters
        ----------
        motor : str
            Metadata key (e.g. ``'mirror.cs_rz'``).
        step_start : int
        step_end : int

        Returns
        -------
        fig, (axs,)
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        rax0 = axs[0].twinx()
        rax1 = axs[1].twinx()
        plt.subplots_adjust(hspace=0.3)

        baseline_m, final_m = [], []
        baseline_cx, final_cx = [], []
        step_idx = []

        for i, scan in enumerate(self.scans):
            bm = fm = bcx = fcx = None
            last_idx = max(s.step_index for s in scan.steps)
            end_ix = last_idx if step_end == -1 else step_end
            for s in scan.steps:
                if s.step_index == step_start:
                    bm = _extract_metadata_value(s.metadata.get(motor, 0))
                    bcx = (s.beam_properties.get('centroid_x', 0)
                           if s.beam_properties else 0)
                if s.step_index == end_ix:
                    fm = _extract_metadata_value(s.metadata.get(motor, 0))
                    fcx = (s.beam_properties.get('centroid_x', 0)
                           if s.beam_properties else 0)
            if None in (bm, fm):
                continue
            baseline_m.append(bm)
            final_m.append(fm)
            baseline_cx.append(bcx)
            final_cx.append(fcx)
            step_idx.append(i)

        bm_arr = np.array(baseline_m)
        fm_arr = np.array(final_m)
        bcx_arr = np.array(baseline_cx)
        fcx_arr = np.array(final_cx)

        init_m = bm_arr[0]
        init_cx = bcx_arr[0]

        motor_delta = (fm_arr - init_m) - (bm_arr - init_m)
        cx_delta = (fcx_arr - init_cx) - (bcx_arr - init_cx)

        motor_label = motor.replace('.', ' ').upper()

        axs[0].plot(step_idx, motor_delta, marker='o', color='blue',
                    label=motor_label)
        rax0.plot(step_idx, cx_delta, marker='o', color='green',
                  label='Centroid X (px)')

        l1, lb1 = axs[0].get_legend_handles_labels()
        l2, lb2 = rax0.get_legend_handles_labels()
        axs[0].legend(l1 + l2, lb1 + lb2, loc='best')
        axs[0].grid(True)
        axs[0].set_xlabel('Scan Index')
        axs[0].set_ylabel(motor_label)
        rax0.set_ylabel('Centroid X Change (px)')
        axs[0].set_title(f'{motor_label} Change vs. Scan Index')

        axs[1].plot(step_idx, np.cumsum(motor_delta), marker='o',
                    color='blue', label=f'Cumulative {motor_label}')
        rax1.plot(step_idx, np.cumsum(cx_delta), marker='o',
                  color='green', label='Cumulative Centroid X (px)')

        l1, lb1 = axs[1].get_legend_handles_labels()
        l2, lb2 = rax1.get_legend_handles_labels()
        axs[1].legend(l1 + l2, lb1 + lb2, loc='best')
        axs[1].grid(True)
        axs[1].set_xlabel('Scan Index')
        axs[1].set_ylabel(f'Cumulative {motor_label}')
        rax1.set_ylabel('Cumulative Centroid X (px)')
        axs[1].set_title(f'Cumulative {motor_label} Change vs. Scan Index')

        plt.tight_layout()
        return fig, axs

    def __repr__(self):
        return (f"<DataSet: {len(self.scans)} × {self.scan_type}, "
                f"var={self.scan_variable}>")


# ===========================================================================
#  Old-style API  (preserved for backward compatibility)
# ===========================================================================
#  These functions operate on raw nested dicts as before.  New code should
#  use the DataStep / DataScan / DataSet classes defined above.

def files_in_directory(wdir, pattern):
    """List files in a directory matching a regex pattern."""
    return DataSet.files_in_directory(wdir, pattern)


def h5_to_dict(filename):
    """Read an HDF5 file into a nested dict."""
    return DataSet.h5_to_dict(filename)


def dataset_from_h5_files(files):
    """Read multiple HDF5 files into a ``dataset`` nested dict."""
    dataset = {}
    for filename in files:
        key = re.sub('pass', '',
                     os.path.basename(filename).split('-')[0].split('_')[-1])
        dataset[key] = h5_to_dict(filename)
    return dataset


def _get_dev_val(dataset, nscan, dev):
    """Helper to extract a device value from the dataset."""
    val = dataset[nscan]['attrs'].get(dev)
    if isinstance(val, (np.ndarray, list)):
        val = val[0]
    return val


def get_scan_data(dataset, variable, observable):
    """Extract observable and variable values across all scans."""
    ndset = list(dataset.keys())
    DataScan = []
    for ns in ndset:
        scanlist = list(dataset[ns].keys())
        obs_set = []
        var_set = []
        for nscan in scanlist:
            obsval = _get_dev_val(dataset[ns], nscan, observable)
            obs_set.append(obsval)
            varval = _get_dev_val(dataset[ns], nscan, variable)
            var_set.append(varval)
        DataScan.append((obs_set, var_set))
    return DataScan


def _get_variable_metadata(data, dev_motor):
    """Extract variable metadata from a single step dict."""
    device, motor = dev_motor.split('.')
    try:
        if data['attrs'].get(dev_motor) is not None:
            meta = data['attrs'].get(dev_motor)
        elif data.get(device, None) is not None:
            meta = data[device]['attrs'].get(motor)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            f"Could not extract scanned variable metadata for {dev_motor}"
        ) from err
    return meta


def observable_statistics(dataset, observable):
    """Calculate statistics of an observable across scans."""
    yscans = []
    for DataScan in dataset.values():
        _, _, xvals, yvals, _ = observable_data(DataScan, observable)
        yscans.append(yvals)
    yscans = np.array(yscans)
    return {
        'xval':    xvals,
        'mean':    np.mean(yscans, axis=0),
        'median':  np.median(yscans, axis=0),
        'std_dev': np.std(yscans, axis=0),
    }


def observable_data(DataScan, observable, droi=4):
    """Extract observable behaviour across steps in a single scan."""
    first = list(DataScan.values())[0]
    attrs = first['attrs']
    if attrs.get('scan_name') == 'slit':
        device = attrs['scan_device']
        dev_motor = f"{device}"
    else:
        motor = attrs['scan_motor']
        device = attrs['scan_device']
        dev_motor = f"{device}.{motor}"

    if observable == 'centroid':
        steps, xvals, cents, sigmas = beam_centroid(DataScan, dev_motor, droi)
        centroids = [cents[:, 0], cents[:, 1]]
        sigmas = [sigmas[:, 0], sigmas[:, 1]]
        return motor, steps, xvals, centroids, sigmas

    if observable == 'fwhm':
        fwhms = beam_fwhm(DataScan, dev_motor, droi)
        steps = np.array(list(fwhms.keys()))
        xvals = np.array([fwhms[step][0] for step in steps])
        cvalues = np.array([fwhms[step][1] for step in steps])
        fwhms_out = [cvalues[:, 0], cvalues[:, 1]]
        return motor, steps, xvals, fwhms_out, None

    if observable == 'intensity':
        intensities = beam_intensity(DataScan, dev_motor, droi)
        steps = np.array(list(intensities.keys()))
        xvals = np.array([intensities[step][0] for step in steps])
        cvalues = np.array([intensities[step][1] for step in steps])
        ints = [cvalues[:, 0], cvalues[:, 1], cvalues[:, 2]]
        sigmas = [np.sqrt(i) for i in ints]
        return motor, steps, xvals, ints, sigmas

    steps, xval, yval = [], [], []
    for step, DataStep in DataScan.items():
        steps.append(int(step.split('-')[-1]))
        xmeta = _get_variable_metadata(DataStep, dev_motor)
        ymeta = DataStep['attrs'].get(f"{device}.{observable}")
        xval.append(float(xmeta[0]) if isinstance(xmeta, (list, np.ndarray)) else float(xmeta))
        yval.append(float(ymeta[0]) if isinstance(ymeta, (list, np.ndarray)) else float(ymeta))

    return motor, np.array(steps), np.array(xval), [np.array(yval)], None


def beam_from_scan(DataScan, dev_motor, droi=4, analysis_mode='qck'):
    """Return beam instances from a scan dict."""
    beam_instances = {}
    for step, data in DataScan.items():
        st = int(step.split('-')[-1])
        xval = float(_get_variable_metadata(data, dev_motor)[0])
        img = data['dvf_B1']['data']
        xedges = np.arange(img.shape[0] + 1)
        yedges = np.arange(img.shape[1] + 1)
        ana = Histogram2DAnalyzer(img, xedges=xedges, yedges=yedges, droi=droi)
        if not ana.beam_visible:
            continue
        ana.analyze(analysis_mode)
        beam_instances[st] = [xval, ana]
    return beam_instances


def beam_centroid(datascan, dev_motor, droi=4, analysis_mode='qck'):
    """Return centroids from a scan dict."""
    beam_instances = beam_from_scan(datascan, dev_motor, droi, analysis_mode)
    steps, xvals, centrs, sigmas = [], [], [], []
    for step, values in beam_instances.items():
        steps.append(step)
        xvals.append(values[0])
        ana = values[1]
        hprm = getattr(ana, f"hprm_{analysis_mode}")
        centrs.append([hprm['mux'], hprm['muy']])
        sigmas.append([hprm['sigx'], hprm['sigy']])
    return np.array(steps), np.array(xvals), np.array(centrs), np.array(sigmas)


def beam_fwhm(datascan, dev_motor, droi=4, analysis_mode='qck'):
    """Return FWHMs from a scan dict."""
    fwhms = {}
    beam_instances = beam_from_scan(datascan, dev_motor, droi, analysis_mode)
    for st in beam_instances:
        xval = beam_instances[st][0]
        ana = beam_instances[st][1]
        hprm = getattr(ana, f"hprm_{analysis_mode}")
        fx, fy = hprm['fwhmx'], hprm['fwhmy']
        fwhms[st] = [xval, [fx, fy]]
    return fwhms


def beam_intensity(datascan, dev_motor, droi=4, analysis_mode='qck'):
    """Return intensities from a scan dict."""
    beam_instances = beam_from_scan(datascan, dev_motor, droi, analysis_mode)
    intensities = {}
    for step in beam_instances:
        dstep = f"scan-{step:04d}"
        exptime = datascan[dstep]['dvf_B1']['attrs']['expo_time']
        xval = beam_instances[step][0]
        ana = beam_instances[step][1]
        hprm = getattr(ana, f"hprm_{analysis_mode}")
        img = ana.img
        cx, cy = hprm['mux'], hprm['muy']
        fx, fy = hprm['fwhmx'], hprm['fwhmy']
        peak = np.mean(img[cy - droi:cy + droi + 1,
                           cx - droi:cx + droi + 1])
        mask = img > (peak / 2)
        area_mask = np.sum(mask)
        img_masked = np.where(mask, img, 0)
        area_img_masked = np.sum(img_masked)
        intensity_by_mask = (area_img_masked / (area_mask * exptime)
                             if area_mask != 0 else 0)
        peak /= exptime
        peak_fwhm_norm = (peak / (fx * fy) if fx * fy != 0 else 0)
        intensities[step] = [xval, [peak, intensity_by_mask, peak_fwhm_norm]]
    return intensities


def correlate(a, b):
    """Normalised cross-correlation between two 1-D arrays."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    norm = np.std(a) * np.std(b) * len(a)
    return np.correlate(a, b, mode='full') / norm


# ===========================================================================
#  Old-style plotting functions  (preserved for backward compatibility)
# ===========================================================================

def dataset_plot(ax, xvals, yvals, datakey, observable, motor,
                 first_item=0, last_item=None, annotate_points=True):
    """Plot observable vs. step for a single dataset on *ax*."""
    if annotate_points:
        for i, (tx, yv) in enumerate(zip(
                xvals[first_item:last_item], yvals[first_item:last_item])):
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


def plot_double_observable(axs, nrow, dataset, observable, observables,
                           first_item=0, last_item=None, droi=4):
    """Plot two-component observables in separate subplots."""
    for key, data in dataset.items():
        motor, steps, xvals, yvals, sigmas = observable_data(data, observable,
                                                             droi=droi)
        dataset_plot(axs[nrow, 0], xvals, yvals[0], key,
                     f"{observable} X", motor, first_item, last_item)
        dataset_plot(axs[nrow, 1], xvals, yvals[1], key,
                     f'{observable} Y', motor, first_item, last_item)
    observables.remove(observable)


def scan_plot(data, observables, first_item=0, last_item=None, droi=8):
    """Plot observables across passes (old-style dict interface)."""
    nobs = len(observables)
    nobs += sum(1 for obs in ['centroid', 'fwhm', 'intensity']
                if obs in observables)
    nrows = max((nobs + 1) // 2, 1)
    ncols = 2 if nobs > 1 else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(10 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axs = [axs]

    nextrow = 0
    for observable in ['centroid', 'fwhm', 'intensity']:
        if observable in observables:
            plot_double_observable(axs, nextrow, data, observable,
                                   observables, first_item, last_item, droi)
            nextrow += 1

    for idx, observable in enumerate(observables):
        if nrows > 1 and ncols > 1:
            nr, nc = divmod(idx + nextrow * ncols, 2)
            ax = axs[nr, nc]
        else:
            ax = axs[idx + nextrow]
        for key, dataset_item in data.items():
            motor, steps, xvals, yvals, sigmas = observable_data(
                dataset_item, observable, droi=droi)
            for yval in yvals:
                dataset_plot(ax, xvals, yval, key, observable, motor,
                             first_item, last_item)
    plt.show()


def centroid_plot(data, steppass, wdir='.', save_fmt='gif'):
    """Plot beam images and centroids (old-style dict interface)."""
    motor, steps, xval, (cx_all, cy_all), _ = observable_data(data, 'centroid')
    images = [(f'step-{step:04d}',
               data[f'step-{step:04d}']['dvf_B1']['data'])
              for step in steps]
    fig, (ax_img, ax_cx, ax_cy) = plt.subplots(1, 3, figsize=(24, 5))
    im = ax_img.imshow(images[0][1], cmap='viridis', animated=True)
    plt.colorbar(im, ax=ax_img, label='Intensity')
    ax_img.set_xlabel('Pixel X')
    ax_img.set_ylabel('Pixel Y')
    img_title = ax_img.set_title('')
    ax_cx.plot(xval, cx_all, marker='o', color='steelblue', label='Centroid X')
    marker_pt, = ax_cx.plot([], [], 'ro', markersize=10, label='Current')
    ax_cx.set_xlabel(motor.capitalize())
    ax_cx.set_ylabel('Centroid X (px)')
    ax_cx.set_title('Centroid X')
    ax_cx.legend()
    ax_cx.grid(True)
    ax_cy.plot(xval, cy_all, marker='o', color='orange', label='Centroid Y')
    marker_pt_y, = ax_cy.plot([], [], 'ro', markersize=10, label='Current')
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
        anim.save(f'{wdir}/beam_xy_pass_{steppass}.mp4',
                  writer='ffmpeg', fps=2, dpi=300)
    plt.close()
    ipydisplay(HTML(anim.to_jshtml()))


def fwhm_plot(dataset, steppass, wdir='.', save_fmt='gif'):
    """Plot beam images and FWHMs (old-style dict interface)."""
    motor, step_nums, xval, (fx_all, fy_all) = observable_data(dataset, 'fwhm')
    images = [(f'step-{step:04d}',
               dataset[f'step-{step:04d}']['dvf_B1']['data'])
              for step in step_nums]
    fig, (ax_img, ax_fx, ax_fy) = plt.subplots(1, 3, figsize=(24, 5))
    im = ax_img.imshow(images[0][1], cmap='viridis', animated=True)
    plt.colorbar(im, ax=ax_img, label='Intensity')
    ax_img.set_xlabel('Pixel X')
    ax_img.set_ylabel('Pixel Y')
    img_title = ax_img.set_title('')
    ax_fx.plot(xval, fx_all, marker='o', color='steelblue', label='FWHM X')
    marker_pt, = ax_fx.plot([], [], 'ro', markersize=10, label='Current')
    ax_fx.set_xlabel(motor.capitalize())
    ax_fx.set_ylabel('FWHM X (px)')
    ax_fx.set_title('FWHM X')
    ax_fx.legend()
    ax_fx.grid(True)
    ax_fy.plot(xval, fy_all, marker='o', color='orange', label='FWHM Y')
    marker_pt_y, = ax_fy.plot([], [], 'ro', markersize=10, label='Current')
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
        anim.save(f'{wdir}/beam_xy_pass_{steppass}_fwhm.mp4',
                  writer='ffmpeg', fps=2, dpi=300)
    plt.close()
    ipydisplay(HTML(anim.to_jshtml()))


def centroid_x_delta_plot(dataset, motor, step_start=0, step_end=-1):
    """Motor change and centroid-X change across passes."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    rax0 = axs[0].twinx()
    rax1 = axs[1].twinx()
    plt.subplots_adjust(hspace=0.3)
    baseline_motor, final_motor, step_idx = [], [], []
    baseline_cx, final_cx = [], []
    for data in dataset.values():
        baseline_motor.append(data[f'step-{step_start:04d}']['attrs'][motor][0])
        final_motor.append(data[f'step-{step_end:04d}']['attrs'][motor][0])
        step_idx.append(len(step_idx))
        steps, xvals, centroid, sigmas = beam_centroid(data, motor)
        baseline_cx.append(centroid[step_start][0])
        final_cx.append(centroid[step_end][0])
    init_motor = baseline_motor[0]
    init_cx = baseline_cx[0]
    bm = np.array(baseline_motor) - init_motor
    fm = np.array(final_motor) - init_motor
    bc = np.array(baseline_cx) - init_cx
    fc = np.array(final_cx) - init_cx
    motor_delta = fm - bm
    cx_delta = fc - bc
    ml = motor.replace('.', ' ').upper()
    axs[0].plot(step_idx, motor_delta, marker='o', color='blue', label=ml)
    rax0.plot(step_idx, cx_delta, marker='o', color='green',
              label='Centroid X Change (px)')
    l1, lb1 = axs[0].get_legend_handles_labels()
    l2, lb2 = rax0.get_legend_handles_labels()
    axs[0].legend(l1 + l2, lb1 + lb2, loc='best')
    axs[0].grid(True)
    axs[0].set_xlabel('Step Index')
    axs[0].set_ylabel(ml)
    rax0.set_ylabel('Centroid X Change (px)')
    axs[0].set_title(f'{ml} Change vs. Scan Index')
    axs[1].plot(step_idx, np.cumsum(motor_delta), marker='o', color='blue',
                label=f'Cumulative {ml}')
    rax1.plot(step_idx, np.cumsum(cx_delta), marker='o', color='green',
              label='Cumulative Centroid X Change')
    l1, lb1 = axs[1].get_legend_handles_labels()
    l2, lb2 = rax1.get_legend_handles_labels()
    axs[1].legend(l1 + l2, lb1 + lb2, loc='best')
    axs[1].grid(True)
    axs[1].set_xlabel('Step Index')
    axs[1].set_ylabel(f'Cumulative {ml}')
    rax1.set_ylabel('Cumulative Centroid X Change (px)')
    axs[1].set_title(f'Cumulative {ml} Change vs. Step Index')
    plt.show()


def caustic_analysis(filename, filedir):
    """Perform caustic analysis on a single HDF5 file."""
    filepath = os.path.join(filedir, filename)
    f = h5py.File(name=filepath, mode='r')
    positions = [f[scaname]['dvf_B1'].attrs['z_pos'] for scaname in f]
    caustic3d = []
    for scaname in f:
        img = np.array(f[scaname]['dvf_B1'])
        caustic3d.append(img)
    causticx = np.sum(caustic3d, axis=1).T
    causticy = np.sum(caustic3d, axis=2).T
    fwhmsx = [utils.full_width(data=profilex, coords=positions)[0]
              for profilex in causticx.T]
    fwhmsy = [utils.full_width(data=profiley, coords=positions)[0]
              for profiley in causticy.T]
    causticx_params, _ = curve_fit(utils.caustic_func, positions, fwhmsx,
                                   p0=None)
    causticy_params, _ = curve_fit(utils.caustic_func, positions, fwhmsy,
                                   p0=None)
    resultsx = utils.caustic_processing(causticx_params, positions)
    resultsy = utils.caustic_processing(causticy_params, positions)
    f.close()
    return resultsx, resultsy
