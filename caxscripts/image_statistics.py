"""Module for generating and analyzing 2D distribution histograms.

All functionality lives inside Histogram2DAnalyzer:

  Construction
  - Histogram2DAnalyzer(img, x_bin_edges, y_bin_edges)
            — from an existing histogram.
  - Histogram2DAnalyzer.from_gaussian(...)
            — generate a Gaussian histogram.

  Quick Analysis (run on init)
  - compute_quick()     — compute quick beam params
                        (centroid, FWHM visibility).
  - beam_visible        — attribute: whether beam is visible.
  - hprm_qck            — attribute: quick parameters dict.

  Full Analysis
  - compute_moments(img)   — weighted moments and principal-axis parameters.
  - fit(hprm)              — 2D Gaussian fit via nonlinear least squares.
  - compute_threshold()    — Kapur entropy thresholding within the ROI.

  ROI
  - compute_roi()          — rebuild the projection-derived ROI mask.
  - roi_mask               — attribute: boolean ROI mask.
  - roi_bounds             — attribute: (xmin, xmax, ymin, ymax) for plotting.

  Display
  - print_stats(hprm)
  - plot(hprm, ...)        — shows and zooms to the ROI by default.
  - plot_entropy(entropies, bin_edges, optimal_threshold)

  Orchestration
  - analyze(mode)        — run analysis pipeline;
                           mode in ('quick', 'moments', 'fit', 'all').
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------

DEFAULT_MEAN = [0, 0]
DEFAULT_COV = [[1, 0], [0, 1]]
DEFAULT_HIST_RANGE = [[-8, 8], [-8, 8]]
COLORS = {"in": "white", "mid": "green", "out": "orange"}
ELLIPSES = ((1, COLORS["in"]), (2, COLORS["mid"]), (3, COLORS["out"]))
ELLIPSES_TITLE = "2D Histogram with Ellipses"

# Threshold for peak-to-average ratio acceptance of image.
# PEAK2AVG_THRESHOLD = 100
PEAK2AVG_THRESHOLD = 5

# FWHM <-> sigma conversion: FWHM = f2sig * sigma
FWHM2SIG = 2 * np.sqrt(2 * np.log(2))


# ---------------------------------------------------------------------------
# Analysis pipeline class.
# ---------------------------------------------------------------------------


class Histogram2DAnalyzer:
    """Self-contained analysis pipeline for a single 2D histogram.

    Stores the image, bin edges, and intermediate results (moments,
    threshold) as instance attributes.

    Usage::

        # From an existing histogram:
        ana = Histogram2DAnalyzer(hist, x_bin_edges, y_bin_edges)

        # Or generate a Gaussian test histogram directly:
        ana = Histogram2DAnalyzer.from_gaussian(bins=80, size=200_000)
        ana.add_noise()

        hprm = ana.compute_moments()
        ana.print_stats()
        fig, ax = ana.plot()
        plt.show()

        entropies, bin_edges, thr, nbins = ana.compute_threshold()
        fig = ana.plot_entropy(entropies, bin_edges, thr)
        plt.show()

        hprm = ana.compute_moments(img=ana.img_thresholded)
        hprm = ana.fit()
    """

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------

    def __init__(
        self,
        img,
        droi=4,
        x_bin_edges=None,
        y_bin_edges=None,
        roi_sigma=5.0,
    ):
        """Initialise a 2D histogram with optional physical bin edges.

        Arguments:
            img: 2D histogram array.
            x_bin_edges: optional uniform x edges; defaults to bin-index
                coordinates from 0 to img.shape[0].
            y_bin_edges: optional uniform y edges; defaults to bin-index
                coordinates from 0 to img.shape[1].
            droi: nonnegative integer radius, in bins, for local ROIs.
            roi_sigma: half-width of the analysis ROI in projection sigmas.
                Defaults to 3.0, i.e. a 3-sigma rectangular ROI.

        Runs compute_quick() automatically to populate beam_visible
        and hprm_qck.
        """
        self.img = np.asarray(img, dtype=float)
        if self.img.ndim != 2 or min(self.img.shape) == 0:
            raise ValueError("img must be a non-empty 2D array.")
        if not isinstance(droi, (int, np.integer)) or droi < 0:
            raise ValueError("droi must be a nonnegative integer number of bins.")
        try:
            roi_sigma = float(roi_sigma)
        except (TypeError, ValueError) as err:
            raise ValueError("roi_sigma must be a positive finite number.") from err
        if not np.isfinite(roi_sigma) or roi_sigma <= 0:
            raise ValueError("roi_sigma must be a positive finite number.")
        self.droi              = droi
        self.roi_sigma         = roi_sigma
        self.roi_mask          = None
        self.roi_bounds        = None
        self.beam_visible      = None  # set by compute_quick()
        # quick params: cx, cy, fwhm_x, fwhm_y, sig_x, sig_y
        self.hprm_project      = None
        self.hprm_momenta      = None  # momenta params
        self.hprm_fitting      = None  # fitting params
        self.hprm              = None
        self.optimal_threshold = np.max(img) * 0.1
        self.img_thresholded   = None
        self.data              = None
        self.pixel_size        = 1
        self._bin_centers(x_bin_edges, y_bin_edges)

        # Run quick analysis automatically
        self.parameters_from_projections()

    # ------------------------------------------------------------------
    # Quick analysis block (run on init).
    # ------------------------------------------------------------------

    def _compute_beam_visibility(self, cx, cy):
        """Compute whether the beam is visible based on peak-to-average ratio.

        Args:
            cx: x-coordinate of the beam centroid in edge-coordinate units.
            cy: y-coordinate of the beam centroid in edge-coordinate units.

        Returns:
            bool: True if the beam is considered visible.
        """
        ix = self._nearest_bin_index(cx, "x")
        iy = self._nearest_bin_index(cy, "y")
        roi = self.img[
            max(0, ix - self.droi):min(self.img.shape[0], ix + self.droi + 1),
            max(0, iy - self.droi):min(self.img.shape[1], iy + self.droi + 1),
        ]
        roi_avg = np.mean(roi)
        mean = np.mean(self.img)
        ratio = roi_avg / mean if mean != 0 else 0
        return ratio >= PEAK2AVG_THRESHOLD

    def _centroid_from_projection(self):
        """Return the peak-projection centroid in bin-edge coordinates.

        Returns:
            np.array: [cx, cy] bin-center coordinates.
        """
        # image is being transposed before being attributed to the class
        # then, its shape is (nx, ny) and the x projection is along axis=1,
        # while the y projection along axis=0
        cx_sum = np.sum(self.img, axis=1)
        cy_sum = np.sum(self.img, axis=0)
        def smooth_projection(projection):
            window = min(21, projection.size if projection.size % 2 else projection.size - 1)
            if window < 3:
                return projection
            return savgol_filter(projection, window_length=window, polyorder=2)

        ix = np.argmax(smooth_projection(cx_sum))
        iy = np.argmax(smooth_projection(cy_sum))
        return np.array([self.x_bin_centers[ix], self.y_bin_centers[iy]])

    def _fwhm_from_projection(self, cx, cy, img=None):
        """Estimate projection FWHM in the coordinate units of the edges.

        Args:
            cx: x-coordinate of the beam centroid in edge-coordinate units.
            cy: y-coordinate of the beam centroid in edge-coordinate units.
            img: optional nonnegative weight array to analyze instead of
                self.img (e.g. self.img_thresholded).

        Returns:
            np.array: [fwhm_x, fwhm_y] in bin-edge coordinate units.
        """
        img = self.img if img is None else np.asarray(img, dtype=float)
        ix = self._nearest_bin_index(cx, "x")
        iy = self._nearest_bin_index(cy, "y")

        def profile_fwhm(profile, centers):
            peak_idx = int(np.argmax(profile))
            level = 0.5 * profile[peak_idx]
            left = np.flatnonzero(profile[:peak_idx] <= level)
            right = np.flatnonzero(profile[peak_idx + 1:] <= level)
            if left.size:
                i = left[-1]
                x0, x1 = centers[i], centers[i + 1]
                y0, y1 = profile[i], profile[i + 1]
                xleft = (x1 if y1 == y0 else
                         x0 + (level - y0) * (x1 - x0) / (y1 - y0))
            else:
                xleft = centers[0]
            if right.size:
                i = peak_idx + 1 + right[0]
                x0, x1 = centers[i - 1], centers[i]
                y0, y1 = profile[i - 1], profile[i]
                xright = (x0 if y1 == y0 else
                          x0 + (level - y0) * (x1 - x0) / (y1 - y0))
            else:
                xright = centers[-1]
            return max(0.0, xright - xleft)

        fwhm_x = profile_fwhm(img[:, iy], self.x_bin_centers)
        fwhm_y = profile_fwhm(img[ix, :], self.y_bin_centers)
        return np.array([fwhm_x, fwhm_y])

    @staticmethod
    def sigma_to_fwhm(sig):
        """Convert sigma to FWHM.

        Args:
            sig: sigma value(s).

        Returns:
            FWHM value(s).
        """
        return sig * FWHM2SIG

    @staticmethod
    def fwhm_to_sigma(fwhm):
        """Convert FWHM to sigma.

        Args:
            fwhm: FWHM value(s).

        Returns:
            sigma value(s).
        """
        return fwhm / FWHM2SIG

    @staticmethod
    def adjust_angle(theta):
        """Adjust angle to be within [-pi, pi].

        Args:
            theta: angle in radians.

        Returns:
            Adjusted angle in radians.
        """
        theta_wrapped = (theta + np.pi) % (2 * np.pi)
        if theta_wrapped > np.pi:
            theta_wrapped -= 2 * np.pi
        return theta_wrapped

    def parameters_from_projections(self):
        """Compute beam parameters from projections automatically.

        Sets self.beam_visible and self.hprm_project if the beam is visible.

        Args:
            img: optional array to analyze instead of self.img
                (e.g. self.img_thresholded).

        Returns:
            dict or None: hprm_qck if visible, None otherwise.
        """
        if self.img.ndim != 2:
            raise ValueError("img must be a 2D array.")
        nx, ny = self.img.shape
        if self.x_bin_edges.size != nx + 1:
            raise ValueError("x_bin_edges length must be img.shape[0] + 1.")
        if self.y_bin_edges.size != ny + 1:
            raise ValueError("y_bin_edges length must be img.shape[1] + 1.")

        cx, cy = self._centroid_from_projection()
        self.beam_visible = self._compute_beam_visibility(cx, cy)

        if not self.beam_visible:
            self.hprm_project = None
            self.compute_roi()
            import warnings
            warnings.warn("Beam not visible; skipping quick analysis.",
                          stacklevel=2)
            return None

        fwhms = self._fwhm_from_projection(cx, cy)
        sigx  = self.fwhm_to_sigma(fwhms[0])
        sigy  = self.fwhm_to_sigma(fwhms[1])
        self.hprm_project = {
            "mux"       : cx,
            "muy"       : cy,
            "sigx"      : sigx,
            "sigy"      : sigy,
            "fwhmx"     : fwhms[0],
            "fwhmy"     : fwhms[1],
            # no cov. in quick estimate
            "cov"       : np.diag([sigx**2, sigy**2]),
            "sig_major" : max(sigx, sigy),
            "sig_minor" : min(sigx, sigy),
            # ellipse aligned to axes
            "theta"     : 0.5*np.pi*float(sigx < sigy),
            "amplitude" : self._amplitude_estimate(),
            "threshold" : np.min(self.img),
            # identity for quick estimate
            "evecs"     : np.array([[1, 0], [0, 1]]),
            "x_bin_centers"  : self.x_bin_centers,
            "y_bin_centers"  : self.y_bin_centers,

        }
        self.compute_roi()
        return self.hprm_project

    def compute_roi(self):
        """Rebuild the projection-derived rectangular ROI.

        Uses the projection centroid and sigmas from self.hprm_project,
        with half-width self.roi_sigma * sigma on each axis. Stores the
        boolean mask in self.roi_mask and plotting bounds in self.roi_bounds.

        Returns:
            np.ndarray or None: the boolean ROI mask, or None when the
                projection parameters are unavailable.
        """
        self.roi_mask = None
        self.roi_bounds = None
        if self.hprm_project is None:
            return None

        hprm = self.hprm_project
        x0 = hprm["mux"] - self.roi_sigma * hprm["sigx"]
        x1 = hprm["mux"] + self.roi_sigma * hprm["sigx"]
        y0 = hprm["muy"] - self.roi_sigma * hprm["sigy"]
        y1 = hprm["muy"] + self.roi_sigma * hprm["sigy"]

        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")
        roi_mask = (
            (xg >= x0) & (xg <= x1) &
            (yg >= y0) & (yg <= y1)
        )
        if not np.any(roi_mask):
            raise ValueError(
                "ROI does not contain any bin centers; increase roi_sigma."
            )

        self.roi_mask = roi_mask
        self.roi_bounds = (
            max(float(self.x_bin_edges[0]), x0),
            min(float(self.x_bin_edges[-1]), x1),
            max(float(self.y_bin_edges[0]), y0),
            min(float(self.y_bin_edges[-1]), y1),
        )
        return self.roi_mask

    def analyze(self, mode='all', warn=True):
        """Run analysis pipeline.

        Args:
            mode: 'quick' | 'moments' | 'fit' | 'all'. Default 'all'.
            warn: print warning if beam not visible. Default True.

        Returns:
            dict with results based on mode.
        """
        if mode == 'quick':
            return self.parameters_from_projections()

        if not self.beam_visible:
            if warn:
                import warnings
                warnings.warn(
                    f"Beam not visible (mode={mode}); "
                    "skipping advanced analysis.",
                    stacklevel=2
                )
            return None

        result = {}

        if mode in ('momenta', 'all'):
            result['moments'] = self.compute_momenta()

        if mode in ('fitting', 'all'):
            result['fit'] = self.fit()

        return result

    # Obsolete(ish) -- a single function calculating 3 things is 
    # awkward, we must settle on a single definition of intensity.
    # Refer to _amplitude_estimate to get peak count around the 
    # centroid.
    def qck_intensity(self, img, centroid, fwhms, exptime, droi=3):
        """Calculate beam intensity via various normalizations.

        Args:
            img: 2D image array.
            centroid: (cx, cy) coordinates in bin-edge units.
            fwhms: (fwhm_x, fwhm_y).
            exptime: exposure time.
            droi: nonnegative integer ROI radius in bins.

        Returns:
            np.array: [peak, intensity_by_mask, peak_fwhm_norm].
        """
        if not isinstance(droi, (int, np.integer)) or droi < 0:
            raise ValueError("droi must be a nonnegative integer number of bins.")
        cx, cy = centroid
        fx, fy = fwhms
        img = np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")

        ix = self._nearest_bin_index(cx, "x")
        iy = self._nearest_bin_index(cy, "y")
        roi = img[
            max(0, ix - droi):min(img.shape[0], ix + droi + 1),
            max(0, iy - droi):min(img.shape[1], iy + droi + 1),
        ]
        peak = np.mean(roi)
        peak /= exptime
        peak_fwhm_norm = peak / (fx * fy) if fx * fy != 0 else 0

        mask = img > (peak / 2)
        mask_area = np.sum(mask)
        masked_img = np.where(mask, img, 0)
        masked_img_area = np.sum(masked_img)
        intensity_by_mask = (
            masked_img_area / (mask_area * exptime) if mask_area != 0 else 0
        )

        return np.array([peak, intensity_by_mask, peak_fwhm_norm])

    @classmethod
    def from_gaussian(
        cls,
        bins=150,
        size=1_000_000,
        mean=DEFAULT_MEAN,
        cov=DEFAULT_COV,
        hist_range=DEFAULT_HIST_RANGE,
    ):
        """Create an analyzer from a randomly sampled Gaussian distribution.

        Arguments:
            cls: Histogram2DAnalyzer class.
            bins: number of bins along each axis.
            size: number of samples to draw.
            mean: 2-element mean vector.
            cov: 2x2 covariance matrix.
            hist_range: [[xmin, xmax], [ymin, ymax]] histogram range.

        Returns:
            Histogram2DAnalyzer instance.
        """
        rng = np.random.default_rng()
        samples = rng.multivariate_normal(mean=mean, cov=cov, size=size)
        hist, x_bin_edges, y_bin_edges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=bins, range=hist_range
        )
        return cls(hist, x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges)

    # ------------------------------------------------------------------
    # Preprocessing.
    # ------------------------------------------------------------------

    def add_noise(self, noisetype="poisson", level=None):
        """Add noise to self.img in-place.

        Arguments:
            noisetype: 'poisson' (default) or 'gaussian'.
            level: noise scale; defaults to self.img for Poisson.
        """
        rng = np.random.default_rng()
        if noisetype not in ("poisson", "gaussian"):
            raise ValueError("noisetype must be 'poisson' or 'gaussian'.")
        if noisetype == "poisson":
            lam = self.img if level is None else level
            self.img = self.img + rng.poisson(lam, size=self.img.shape)
        else:
            self.img = self.img + rng.normal(
                loc=0, scale=np.sqrt(self.img), size=self.img.shape
            )

    # ------------------------------------------------------------------
    # Private math helpers.
    # ------------------------------------------------------------------

    def _bin_centers(self, x_bin_edges, y_bin_edges):
        """Set uniform bin edges and centers, defaulting to bin coordinates.

        Each missing axis gets edges from 0 to its bin count. Supplied edges
        must be finite, strictly increasing, and uniformly spaced.
        """
        def validate_edges(edges, count, axis):

            # Typically, edges is np.array(None), so the usual is None condition
            # will never fire, although the object IS None. the extra condition
            # ensures things will run smoothly.
            if (edges is None) or (edges == None):
                return np.arange(count + 1, dtype=float)
            try:
                edges = np.asarray(edges, dtype=float)
            except (TypeError, ValueError) as err:
                raise ValueError(f"{axis}_bin_edges must be numeric.") from err
            if edges.ndim != 1 or edges.size != count + 1:
                raise ValueError(
                    f"{axis}_bin_edges must be 1D with {count + 1} entries."
                )
            if not np.all(np.isfinite(edges)):
                raise ValueError(f"{axis}_bin_edges must contain finite values.")
            widths = np.diff(edges)
            if np.any(widths <= 0):
                raise ValueError(f"{axis}_bin_edges must be strictly increasing.")
            if not np.allclose(widths, widths[0], rtol=1e-7, atol=1e-12 * max(1.0, abs(widths[0]))):
                raise ValueError(f"{axis}_bin_edges must be uniformly spaced.")
            return edges

        self.x_bin_edges = validate_edges(x_bin_edges, self.img.shape[0], "x")
        self.y_bin_edges = validate_edges(y_bin_edges, self.img.shape[1], "y")
        self.x_bin_centers = 0.5 * (self.x_bin_edges[:-1] + self.x_bin_edges[1:])
        self.y_bin_centers = 0.5 * (self.y_bin_edges[:-1] + self.y_bin_edges[1:])
        self.x_bin_width = float(self.x_bin_edges[1] - self.x_bin_edges[0])
        self.y_bin_width = float(self.y_bin_edges[1] - self.y_bin_edges[0])

    def _nearest_bin_index(self, coordinate, axis):
        """Map a physical/bin coordinate to the nearest bin-center index."""
        if axis == "x":
            centers = self.x_bin_centers
        elif axis == "y":
            centers = self.y_bin_centers
        else:
            raise ValueError("axis must be 'x' or 'y'.")
        return int(np.argmin(np.abs(centers - coordinate)))

    def _covariance_from_moments(self, weight, useroi=False):
        """Compute the 2x2 covariance matrix from weighted bin-center moments.

        Args:
            weight: 2D weight array (same shape as the histogram).
            useroi: if True, use the projection-derived rectangular ROI
                stored in self.roi_mask.

        Returns:
            (mux, muy): means.
            covmat: 2x2 covariance matrix.
        """
        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")

        if useroi:
            if self.roi_mask is None:
                raise ValueError(
                    "ROI is not available; run compute_roi() first."
                )
            if self.roi_mask.shape != weight.shape:
                raise ValueError(
                    "ROI mask shape must match the weight array shape."
                )
            xg     = xg[self.roi_mask]
            yg     = yg[self.roi_mask]
            weight = weight[self.roi_mask]
        
        wsum = weight.sum()
        if wsum <= 0:
            print(f"WARNING: Total weight must be positive. wsum={wsum}")
            return (None, None), None
        mux = (weight * xg).sum() / wsum
        muy = (weight * yg).sum() / wsum
        dx = xg - mux
        dy = yg - muy
        varx  = (weight * dx * dx).sum() / wsum
        vary  = (weight * dy * dy).sum() / wsum
        covxy = (weight * dx * dy).sum() / wsum
        covmat = np.array([[varx, covxy], [covxy, vary]])
        return (mux, muy), covmat

    def _ellipse_params_from_cov(self, cov):
        """Return principal-axis parameters from a 2x2 covariance matrix.

        Arguments:
            cov: 2x2 covariance matrix.

        Returns:
            sig_major: std along the major axis.
            sig_minor: std along the minor axis.
            theta: rotation angle of the major axis (radians).
            evecs: eigenvectors matrix (columns = principal directions).
        """
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        sig_major = np.sqrt(evals[0])
        sig_minor = np.sqrt(evals[1])
        theta = np.arctan2(evecs[1, 0], evecs[0, 0])
        return sig_major, sig_minor, theta, evecs

    @staticmethod
    def _gaussian_2d(coords, thresh, amplitude, mux, muy, sigx, sigy, covxy):
        """Evaluate a normalized 2D Gaussian at coords = (x, y).

        Arguments:
            coords : tuple (x, y) of coordinate arrays.
            thresh : threshold value.
            amplitude : peak amplitude.
            mux    : mean in x.
            muy    : mean in y.
            sigx   : standard deviation in x.
            sigy   : standard deviation in y.
            covxy  : covariance between x and y.

        Returns:
            Gaussian values at the given coordinates.
        """
        x, y = coords
        cov = np.array([[sigx**2, covxy], [covxy, sigy**2]])
        inv_cov = np.linalg.inv(cov)
        dx = x - mux
        dy = y - muy
        exponent = -0.5 * (
            inv_cov[0, 0] * dx**2
            + 2 * inv_cov[0, 1] * dx * dy
            + inv_cov[1, 1] * dy**2
        )
        # norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
        # return thresh + amplitude * norm * np.exp(exponent)
        return thresh + amplitude * np.exp(exponent)

    def _number_of_bins(self, data=None) -> int:
        """Return a bin count for threshold analysis.

        Uses Freedman-Diaconis as the primary estimate, capped at 1024,
        and falls back to Sturges' rule when FD is smaller. Returns at
        least 2.

        Args:
            data: optional 1D or 2D value array; defaults to self.img.
        """
        values = np.asarray(self.img if data is None else data)
        if values.size == 0:
            raise ValueError("data must contain at least one value.")
        n = values.size
        edgehist = np.histogram_bin_edges(values.ravel(), bins="fd")
        fd_nbins = min(len(edgehist) - 1, 1024)
        sturges_nbins = int(1.0 + np.log2(n))
        return max(fd_nbins, sturges_nbins, 2)

    @staticmethod
    def _kapur_entropy(freq: np.ndarray, k: int) -> float:
        """Kapur cross-class entropy for a histogram split at bin index k.

        Arguments:
            freq: 1D frequency array from np.histogram.
            k: index splitting background (< k) from foreground (>= k).

        Returns:
            Total cross-class entropy (background + foreground).
        """
        freq_b = freq[:k]
        freq_f = freq[k:]
        p_b = freq_b / freq_b.sum() if freq_b.sum() > 0 else np.array([])
        p_f = freq_f / freq_f.sum() if freq_f.sum() > 0 else np.array([])
        h_b = -np.sum(p_b * np.log2(p_b + 1e-10))
        h_f = -np.sum(p_f * np.log2(p_f + 1e-10))
        return h_b + h_f

    # ------------------------------------------------------------------
    # Public pipeline methods.
    # ------------------------------------------------------------------

    def compute_momenta(self, img=None, useroi=True):
        """Compute weighted moments in the bin-edge coordinate units.

        Arguments:
            img: optional array to analyse instead of self.img
                (e.g. self.img_thresholded).
            useroi: if True, use the projection-derived rectangular ROI.

        Returns:
            hprm: dictionary with centroid, sigma, and covariance in the
                bin-edge coordinate units; also stored in self.hprm.
        """
        if not self.beam_visible:
            import warnings
            warnings.warn("Beam not visible; skipping momenta analysis.",
                          stacklevel=2)
            return None
        
        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.ndim != 2:
            raise ValueError("img must be a 2D array.")
        nx, ny = img.shape
        if self.x_bin_edges.size != nx + 1:
            raise ValueError("x_bin_edges length must be img.shape[0] + 1.")
        if self.y_bin_edges.size != ny + 1:
            raise ValueError("y_bin_edges length must be img.shape[1] + 1.")
        
        (mux, muy), covmat = self._covariance_from_moments(img, useroi=useroi)
        sigx = np.sqrt(covmat[0, 0])
        sigy = np.sqrt(covmat[1, 1])
        sig_major, sig_minor, theta, evecs = self._ellipse_params_from_cov(
            covmat
            )

        self.hprm_momenta = {
            "mux"           : mux,
            "muy"           : muy,
            "cov"           : covmat,
            "sigx"          : sigx,
            "sigy"          : sigy,
            "fwhmx"         : self.sigma_to_fwhm(sigx),
            "fwhmy"         : self.sigma_to_fwhm(sigy),
            "sig_major"     : sig_major,
            "sig_minor"     : sig_minor,
            "theta"         : self.adjust_angle(theta),
            "evecs"         : evecs,
            "amplitude"     : self._amplitude_estimate(img, self.droi),
            "threshold"     : np.min(img),
            "x_bin_centers" : self.x_bin_centers,
            "y_bin_centers" : self.y_bin_centers,
        }
        return self.hprm_momenta

    def _amplitude_estimate(self, img = None, droi=4):
        """Estimate amplitude for Gaussian fitting.

        Arguments:
            img: 2D array to analyze.
            droi: nonnegative integer radius of the local ROI in bins.

        Returns:
            Estimated amplitude value.
        """
        img = self.img if img is None else np.asarray(img)
        if not isinstance(droi, (int, np.integer)) or droi < 0:
            raise ValueError("droi must be a nonnegative integer number of bins.")
        cx, cy = self._centroid_from_projection()
        ix = self._nearest_bin_index(cx, "x")
        iy = self._nearest_bin_index(cy, "y")
        roi = img[
            max(0, ix - droi):min(img.shape[0], ix + droi + 1),
            max(0, iy - droi):min(img.shape[1], iy + droi + 1),
        ]
        return float(np.mean(roi))

    def fit(self, hprm=None, img=None, useroi=True):
        """Fit a 2D Gaussian using the bin-edge coordinate system.

        Uses hprm (or self.hprm, or freshly computed moments) as the
        initial parameter guess.

        Arguments:
            hprm: optional pre-computed moments dict.
            img: optional counts-per-bin array to fit instead of self.img
                (e.g. self.img_thresholded); must have the same shape.
            useroi: if True, fit within the projection-derived rectangular
                ROI stored in self.roi_mask. The supplied hprm is still
                used for the initial parameter guess.

        Returns:
            hprm: fitted parameters dictionary; also stored in self.hprm.
        """
        if not self.beam_visible:
            return None

        if hprm is None:
            hprm = self.hprm_project

        # This is a workaround for Pylance, since the clauses above
        # guarantee hprm is calculated.
        if hprm is None:
            raise ValueError("No initial parameters available for fitting.")

        # Default to self.img if no alternative is provided.
        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.shape != self.img.shape:
            raise ValueError("img must have the same shape as self.img.")

        xg, yg = np.meshgrid(self.x_bin_centers, self.y_bin_centers,
                             indexing="ij")

        # Fit to ROI.
        img_opt = img.copy()
        if useroi:
            if self.roi_mask is None:
                raise ValueError(
                    "ROI is not available; run compute_roi() first."
                )
            xg = xg[self.roi_mask]
            yg = yg[self.roi_mask]
            img_opt = img[self.roi_mask]

        p0 = [
            hprm['threshold'],  # threshold    
            hprm['amplitude'],  # peak    
            hprm["mux"],
            hprm["muy"],
            hprm["sigx"],
            hprm["sigy"],
            hprm["cov"][0, 1],
        ]

        img_opt = img_opt.ravel()

        popt, covfit = curve_fit(
            self._gaussian_2d,
            (xg.ravel(), yg.ravel()),
            img_opt,
            p0,
        )

        sx, sy, sxy = popt[4], popt[5], popt[6]
        covmat = np.array([[sx**2, sxy], [sxy, sy**2]])
        (sig_major, sig_minor,
         theta, evecs) = self._ellipse_params_from_cov(covmat)

        self.hprm_fitting = {
            "mux"           : popt[2],
            "muy"           : popt[3],
            "sigx"          : sx,
            "sigy"          : sy,
            "fwhmx"         : self.sigma_to_fwhm(sx),
            "fwhmy"         : self.sigma_to_fwhm(sy),
            "cov"           : covmat,
            "sig_major"     : sig_major,
            "sig_minor"     : sig_minor,
            "theta"         : self.adjust_angle(theta),
            "evecs"         : evecs,
            "amplitude"     : popt[1],
            "threshold"     : popt[0],
            "x_bin_centers" : self.x_bin_centers,
            "y_bin_centers" : self.y_bin_centers,
        }
        return self.hprm_fitting

    def compute_threshold(self, useroi=True):
        """Run Kapur entropy thresholding on self.img.

        Stores the optimal threshold in self.optimal_threshold and the
        thresholded image in self.img_thresholded. When useroi is True,
        the entropy histogram is built from ROI values only and the
        thresholded image is zero outside the ROI.

        Arguments:
            useroi: if True, restrict thresholding to self.roi_mask.
                Defaults to True. Falls back to the full image with a
                warning when no ROI is available.

        Returns:
            entropies: entropy value for each split index.
            bin_edges: bin edges of the internal 1D histogram.
            optimal_threshold: threshold that maximises cross-class entropy.
            nbins: number of bins used.
        """
        if useroi and self.roi_mask is None:
            import warnings
            warnings.warn(
                "ROI is not available; computing threshold on full image.",
                stacklevel=2,
            )
            useroi = False

        values = self.img[self.roi_mask] if useroi else self.img
        nbins = self._number_of_bins(values)
        freq, bin_edges = np.histogram(values, bins=nbins)
        thrs = bin_edges[:-1]

        entropies = np.array(
            [self._kapur_entropy(freq, k) for k in range(1, len(freq))]
        )
        thr = thrs[np.argmax(entropies)]

        self.optimal_threshold = thr
        if useroi:
            self.img_thresholded = np.where(
                self.roi_mask & (self.img > thr),
                self.img - self.optimal_threshold,
                0.0,
            )
        else:
            self.img_thresholded = np.where(self.img > thr,
                                            self.img - self.optimal_threshold,
                                            0.0)
        return entropies, bin_edges, thr, nbins

    def print_stats(self, hprm=None):
        """Print parameters from any hprm dict.

        Arguments:
            hprm: optional dict to print; auto-selects from
                  hprm_qck -> hprm_mom -> hprm_fit.
        """
        if hprm is None:
            hprm = self.hprm_project or self.hprm_momenta or self.hprm_fitting
        if hprm is None:
            print("No parameters computed.")
            return

        print(f"centroid   = ({hprm['mux']:.4e}, {hprm['muy']:.4e})")
        print(f"sigma x    = {hprm['sigx']:.4e}\nsigma y"
              f"    = {hprm['sigy']:.4e}")

        if 'fwhmx' in hprm:
            print(f"fwhm x     = {hprm['fwhmx']:.4e}\nfwhm y"
                  f"     = {hprm['fwhmy']:.4e}")

        if 'cov' not in hprm:
            return

        print(f"xy cov     = {hprm['cov'][0, 1]:.4e}\n")
        print(
            "principal sigmas:\n"
            f"    sigma minor = {hprm['sig_minor']}\n"
            f"    sigma major = {hprm['sig_major']}\n"
        )
        thetadeg = np.degrees(hprm["theta"])
        print(f"theta = {hprm['theta']:.4e} rad = {thetadeg:.4e} deg\n")
        print(f"cov matrix:\n{hprm['cov']}")

    def _render_histogram(self, fig, ax, colorbar=True):
        """Render self.img over its supplied or default bin-edge coordinates.

        Returns:
            m: the pcolormesh artist.
        """
        m = ax.pcolormesh(
            self.x_bin_edges, self.y_bin_edges, self.img.T,
            shading="auto", cmap="viridis",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.x_bin_edges[0], self.x_bin_edges[-1])
        ax.set_ylim(self.y_bin_edges[0], self.y_bin_edges[-1])
        if colorbar:
            fig.colorbar(m, ax=ax, label="count")
        return m

    def plot(
        self,
        hprm=None,
        fig=None,
        ax=None,
        title=ELLIPSES_TITLE,
        ellipses=ELLIPSES,
        show_ellipse_axes=True,
        center_label="centroid",
        colorbar=True,
        show_roi=True,
        roi_zoom=1.2,
        roi_color="cyan",
        roi_ls="--",
    ):
        """Plot the histogram with sigma ellipses and principal directions.

        Arguments:
            hprm         : moments/fit dict; coordinates and widths use the
                           bin-edge coordinate units.
            fig          : optional figure to plot on; created if None.
            ax           : optional axis to plot on; created if None.
            title        : axes title.
            ellipses     : iterable of (n_sigma, color) pairs.
            show_ellipse_axes : draw principal-axis lines when True.
            center_label : legend label for the centroid marker.
            colorbar     : whether to show a colorbar for the histogram.
            show_roi     : draw the projection-derived ROI and zoom to it.
            roi_zoom     : positive zoom factor applied to the ROI bounds;
                           set to None to draw the ROI without zooming.
            roi_color    : ROI outline color.
            roi_ls       : ROI outline line style.

        Returns:
            fig, ax
        """
        if roi_zoom is not None:
            try:
                roi_zoom = float(roi_zoom)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    "roi_zoom must be a positive finite number or None."
                ) from err
            if not np.isfinite(roi_zoom) or roi_zoom <= 0:
                raise ValueError(
                    "roi_zoom must be a positive finite number or None."
                )

        if hprm is None:
            if self.hprm_momenta is None:
                self.compute_momenta()
            hprm = self.hprm_momenta

        # This is a workaround for Pylance, since the clauses above
        # guarantee hprm is calculated.
        if hprm is None:
            raise ValueError("No initial parameters available for fitting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        self._render_histogram(fig, ax, colorbar=colorbar)

        ax.plot(
            hprm["mux"], hprm["muy"],
            marker="+", color="red",
            label=center_label,
            # ms=10, mew=2,
        )

        for n, color in ellipses:
            ell = Ellipse(
                (hprm["mux"], hprm["muy"]),
                width=2 * n * hprm["sig_major"],
                height=2 * n * hprm["sig_minor"],
                angle=np.rad2deg(hprm["theta"]),
                fill=False, lw=2, color=color, label=f"{n}sigma",
            )
            ax.add_patch(ell)

        if show_ellipse_axes:
            v_major = hprm["evecs"][:, 0]
            v_minor = hprm["evecs"][:, 1]
            l_major = 3 * hprm["sig_major"]
            l_minor = 3 * hprm["sig_minor"]
            ax.plot(
                [hprm["mux"] - l_major * v_major[0],
                 hprm["mux"] + l_major * v_major[0]],
                [hprm["muy"] - l_major * v_major[1],
                 hprm["muy"] + l_major * v_major[1]],
                color="white", ls="--", lw=1,
            )
            ax.plot(
                [hprm["mux"] - l_minor * v_minor[0],
                 hprm["mux"] + l_minor * v_minor[0]],
                [hprm["muy"] - l_minor * v_minor[1],
                 hprm["muy"] + l_minor * v_minor[1]],
                color="white", ls=":", lw=1,
            )

        if show_roi and self.roi_bounds is not None:
            x0, x1, y0, y1 = self.roi_bounds
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, lw=1.5, color=roi_color, ls=roi_ls,
                label="ROI", zorder=5,
            ))

            if roi_zoom is not None:
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                half_w = 0.5 * (x1 - x0) * roi_zoom
                half_h = 0.5 * (y1 - y0) * roi_zoom
                if half_w > 0:
                    xlim = (
                        max(float(self.x_bin_edges[0]), cx - half_w),
                        min(float(self.x_bin_edges[-1]), cx + half_w),
                    )
                    if xlim[1] > xlim[0]:
                        ax.set_xlim(*xlim)
                if half_h > 0:
                    ylim = (
                        max(float(self.y_bin_edges[0]), cy - half_h),
                        min(float(self.y_bin_edges[-1]), cy + half_h),
                    )
                    if ylim[1] > ylim[0]:
                        ax.set_ylim(*ylim)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
        return fig, ax

    def plot_entropy(self, entropies, bin_edges, optimal_threshold):
        """Plot the entropy vs. threshold curve.

        Arguments:
            entropies: array returned by compute_threshold().
            bin_edges: bin edges returned by compute_threshold().
            optimal_threshold: threshold value to mark on the plot.

        Returns:
            fig
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bin_edges[1:-1], entropies, label="Entropy")
        ax.axvline(
            optimal_threshold, color="red", linestyle="--",
            label=f"Optimal Threshold: {optimal_threshold:.4g}",
        )
        ax.set_title("Entropy vs. Threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Entropy")
        ax.legend()
        ax.grid()
        return fig
