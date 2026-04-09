"""Module for generating and analyzing 2D distribution histograms.

All functionality lives inside Histogram2DAnalyzer:

  Construction
  - Histogram2DAnalyzer(img, xedges, yedges)  — from an existing histogram.
  - Histogram2DAnalyzer.from_gaussian(...)     — generate a Gaussian histogram.

  Preprocessing
  - add_noise(noisetype, level)  — add Poisson or Gaussian noise to self.img.

  Analysis pipeline
  - compute_moments(img)   — weighted moments and principal-axis parameters.
  - fit(hprm)              — 2D Gaussian fit via nonlinear least squares.
  - compute_threshold()    — Kapur entropy thresholding.

  Display
  - print_stats(hprm)
  - plot(hprm, ...)
  - plot_entropy(entropies, bin_edges, optimal_threshold)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------

DEFAULT_MEAN = [0, 0]
DEFAULT_COV = [[1, 0], [0, 1]]
DEFAULT_HIST_RANGE = [[-8, 8], [-8, 8]]
COLORS = {"in": "white", "mid": "green", "out": "orange"}
ELLIPSES = ((1, COLORS["in"]), (2, COLORS["mid"]), (3, COLORS["out"]))
ELLIPSES_TITLE = "2D Histogram with Ellipses"


# ---------------------------------------------------------------------------
# Analysis pipeline class.
# ---------------------------------------------------------------------------


class Histogram2DAnalyzer:
    """Self-contained analysis pipeline for a single 2D histogram.

    Stores the image, bin edges, and intermediate results (moments,
    threshold) as instance attributes.

    Usage::

        # From an existing histogram:
        ana = Histogram2DAnalyzer(hist, xedges, yedges)

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

    def __init__(self, img, xedges, yedges):
        """Initialise with a 2D histogram and its bin edges.

        Arguments:
            img: 2D histogram array.
            xedges: 1D array of x bin edges (length img.shape[0] + 1).
            yedges: 1D array of y bin edges (length img.shape[1] + 1).
        """
        self.img               = np.asarray(img,    dtype=float)
        self.xedges            = np.asarray(xedges, dtype=float)
        self.yedges            = np.asarray(yedges, dtype=float)
        self.hprm_mom          = None
        self.hprm_fit          = None
        self.optimal_threshold = np.max(img) * 0.1
        self.img_thresholded   = None

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
        hist, xedges, yedges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=bins, range=hist_range
        )
        return cls(hist, xedges, yedges)

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

    def _bin_centers(self):
        """Return (xcenters, ycenters) computed from the stored edges."""
        xcenters = 0.5 * (self.xedges[:-1] + self.xedges[1:])
        ycenters = 0.5 * (self.yedges[:-1] + self.yedges[1:])
        return xcenters, ycenters

    def _covariance_from_moments(self, weight, xcenters, ycenters):
        """Compute the 2x2 covariance matrix from weighted bin-center moments.

        Arguments:
            weight: 2D weight array (same shape as the histogram).
            xcenters: 1D array of x bin centers.
            ycenters: 1D array of y bin centers.

        Returns:
            covmat: 2x2 covariance matrix.
            (mux, muy): means.
        """
        xg, yg = np.meshgrid(xcenters, ycenters, indexing="ij")
        wsum = weight.sum()
        if wsum <= 0:
            raise ValueError("Total weight must be positive.")
        mux = (weight * xg).sum() / wsum
        muy = (weight * yg).sum() / wsum
        dx = xg - mux
        dy = yg - muy
        varx  = (weight * dx * dx).sum() / wsum
        vary  = (weight * dy * dy).sum() / wsum
        covxy = (weight * dx * dy).sum() / wsum
        covmat = np.array([[varx, covxy], [covxy, vary]])
        return covmat, (mux, muy)

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
    def _gaussian_2d(coords, mux, muy, sigx, sigy, covxy):
        """Evaluate a normalized 2D Gaussian at coords = (x, y).

        Arguments:
            coords : tuple (x, y) of coordinate arrays.
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
        norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
        return norm * np.exp(exponent)

    def _number_of_bins(self):
        """Return a bin count for threshold analysis.

        Uses Freedman-Diaconis as the primary estimate, capped at 1024,
        and falls back to Sturges' rule when FD is smaller. Returns at
        least 2.
        """
        n = self.img.size
        edgehist = np.histogram_bin_edges(self.img.ravel(), bins="fd")
        fd_nbins = min(len(edgehist) - 1, 1024)
        sturges_nbins = int(1.0 + np.log2(n))
        return max(fd_nbins, sturges_nbins, 2)

    @staticmethod
    def _kapur_entropy(freq, k):
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

    def compute_moments(self, img=None):
        """Compute weighted moments and principal-axis info.

        Arguments:
            img: optional array to analyse instead of self.img
                (e.g. self.img_thresholded).

        Returns:
            hprm: moments dictionary; also stored in self.hprm.
        """
        img = self.img if img is None else np.asarray(img, dtype=float)
        if img.ndim != 2:
            raise ValueError("img must be a 2D array.")
        nx, ny = img.shape
        if self.xedges.size != nx + 1:
            raise ValueError("xedges length must be img.shape[0] + 1.")
        if self.yedges.size != ny + 1:
            raise ValueError("yedges length must be img.shape[1] + 1.")

        xcenters, ycenters = self._bin_centers()
        covmat, (mux, muy) = self._covariance_from_moments(
            img, xcenters, ycenters
        )
        sigx = np.sqrt(covmat[0, 0])
        sigy = np.sqrt(covmat[1, 1])
        sig_major, sig_minor, theta, evecs = self._ellipse_params_from_cov(
            covmat
        )

        self.hprm_mom = {
            "mux": mux, "muy": muy,
            "cov": covmat,
            "sigx": sigx, "sigy": sigy,
            "sig_major": sig_major, "sig_minor": sig_minor,
            "theta": theta, "evecs": evecs,
            "xcenters": xcenters, "ycenters": ycenters,
        }
        return self.hprm_mom

    def fit(self, hprm=None, img=None, useroi=True):
        """Fit a 2D Gaussian to self.img via nonlinear least squares.

        Uses hprm (or self.hprm, or freshly computed moments) as the
        initial parameter guess.

        Arguments:
            hprm: optional pre-computed moments dict.
            img: optional array to fit instead of self.img
                (e.g. self.img_thresholded).
            useroi: if True, fit only within a 3-sigma ellipse ROI.

        Returns:
            hprm: fitted parameters dictionary; also stored in self.hprm.
        """
        if hprm is None:
            if self.hprm_mom is None:
                self.compute_moments()
            hprm = self.hprm_mom

        # Default to self.img if no alternative is provided.
        if img is None:
            img = self.img

        xcenters = hprm["xcenters"]
        ycenters = hprm["ycenters"]
        xg, yg = np.meshgrid(xcenters, ycenters, indexing="ij")

        # Normalize to PDF so amplitude matches the normalized Gaussian.
        dx = xcenters[1] - xcenters[0]
        dy = ycenters[1] - ycenters[0]
        img_norm = img / (img.sum() * dx * dy)

        # Fit to ROI.
        if useroi:
            roi_mask = (
                (xg >= hprm["mux"] - 3 * hprm["sigx"]) &
                (xg <= hprm["mux"] + 3 * hprm["sigx"]) &
                (yg >= hprm["muy"] - 3 * hprm["sigy"]) &
                (yg <= hprm["muy"] + 3 * hprm["sigy"])
            )
            xg = xg[roi_mask]
            yg = yg[roi_mask]
            img_norm = img_norm[roi_mask]

        p0 = [
            hprm["mux"],
            hprm["muy"],
            hprm["sigx"],
            hprm["sigy"],
            hprm["cov"][0, 1],
        ]
        popt, _ = curve_fit(
            self._gaussian_2d,
            (xg.ravel(), yg.ravel()),
            img_norm.ravel(),
            p0,
        )

        sx, sy, sxy = popt[2], popt[3], popt[4]
        covmat = np.array([[sx**2, sxy], [sxy, sy**2]])
        (sig_major, sig_minor,
         theta, evecs) = self._ellipse_params_from_cov(covmat)

        self.hprm_fit = {
            "mux"       : popt[0],
            "muy"       : popt[1],
            "sigx"      : sx,
            "sigy"      : sy,
            "cov"       : covmat,
            "sig_major" : sig_major,
            "sig_minor" : sig_minor,
            "theta"     : theta,
            "evecs"     : evecs,
            "xcenters"  : xcenters,
            "ycenters"  : ycenters,
        }
        return self.hprm_fit

    def compute_threshold(self):
        """Run Kapur entropy thresholding on self.img.

        Stores the optimal threshold in self.optimal_threshold and the
        thresholded image in self.img_thresholded.

        Returns:
            entropies: entropy value for each split index.
            bin_edges: bin edges of the internal 1D histogram.
            optimal_threshold: threshold that maximises cross-class entropy.
            nbins: number of bins used.
        """
        nbins = self._number_of_bins()
        freq, bin_edges = np.histogram(self.img, bins=nbins)
        thrs = bin_edges[:-1]

        entropies = np.array(
            [self._kapur_entropy(freq, k) for k in range(1, len(freq))]
        )
        thr = thrs[np.argmax(entropies)]

        self.optimal_threshold = thr
        self.img_thresholded = np.where(self.img > thr, self.img, 0.0)
        return entropies, bin_edges, thr, nbins

    def print_stats(self, hprm=None):
        """Print the moments / fit dictionary.

        Arguments:
            hprm: optional dict to print; defaults to self.hprm_mom.
        """
        if hprm is None:
            print("No parameters provided; using the calculated from moments.")
            if self.hprm_mom is None:
                self.compute_moments()
            hprm = self.hprm_mom
        print(f"mu         = ({hprm['mux']:.4e}, {hprm['muy']:.4e})")
        print(f"sigma x    = {hprm['sigx']:.4e}\nsigma y"
              f"    = {hprm['sigy']:.4e}")
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
        """Render self.img as a pcolormesh on ax.

        Returns:
            m: the pcolormesh artist.
        """
        m = ax.pcolormesh(
            self.xedges, self.yedges, self.img.T,
            shading="auto", cmap="viridis",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.xedges[0], self.xedges[-1])
        ax.set_ylim(self.yedges[0], self.yedges[-1])
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
    ):
        """Plot the histogram with sigma ellipses and principal directions.

        Arguments:
            hprm         : moments/fit dict; defaults to self.hprm_mom.
            fig          : optional figure to plot on; created if None.
            ax           : optional axis to plot on; created if None.
            title        : axes title.
            ellipses     : iterable of (n_sigma, color) pairs.
            show_ellipse_axes : draw principal-axis lines when True.
            center_label : legend label for the centroid marker.
            colorbar     : whether to show a colorbar for the histogram.

        Returns:
            fig, ax
        """
        if hprm is None:
            print("No parameters provided; using the calculated from moments.")
            if self.hprm_mom is None:
                self.compute_moments()
            hprm = self.hprm_mom

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
                angle=hprm["theta"] * 180 / np.pi,
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
