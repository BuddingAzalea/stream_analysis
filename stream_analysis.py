from scipy.ndimage import gaussian_filter1d

def measure_stream_LengthWidth(
    phi1: np.ndarray,
    phi2: np.ndarray,
    method: str = "quantile",         # "quantile" or "kde"
    density_frac: float = 0.95,
    nbins: int = 100,                 # number of bins per stream (KDE only)
    smoothing_sigma: float = 1.0,     # Gaussian σ in bin‐units (KDE only)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure stream length (along phi1) and width (along phi2), for one or many streams.

    You can choose:
    - "quantile": length and width are simple [(1−f)/2, (1+f)/2] quantiles.
    - "kde": length is the span of the phi1 region enclosing `density_frac`
             using a per-stream smoothed histogram (approximate KDE),
             and width is the quantile width of phi2 within that region.

    Parameters
    ----------
    phi1 : np.ndarray, shape (N,) or (S, N)
        Along-stream coordinates (degrees or radians).
    phi2 : np.ndarray, shape (N,) or (S, N)
        Across-stream coordinates.
    method : {"quantile","kde"}
        Which estimator to use.
    density_frac : float
        Fraction of total particles to enclose (e.g. 0.95).
    nbins : int
        Number of bins for the per-stream histogram (KDE mode).
    smoothing_sigma : float
        σ of the Gaussian filter, in bin units (KDE mode).

    Returns
    -------
    lengths : np.ndarray of shape (S,) or scalar
        Enclosed span in phi1.
    widths  : np.ndarray of shape (S,) or scalar
        Span in phi2, measured as quantile width within the KDE-defined phi1 region.
    """
    single = (phi1.ndim == 1)
    phi1_arr = np.atleast_2d(phi1)
    phi2_arr = np.atleast_2d(phi2)
    S, N = phi1_arr.shape
    assert phi2_arr.shape == (S, N), "phi1/phi2 must match shapes"

    if method == "quantile":
        # -------------------------
        # 1) Compute quantile-based length & width
        # -------------------------
        lo = (1 - density_frac) / 2
        hi = 1 - lo
        q1 = np.quantile(phi1_arr, [lo, hi], axis=1)  # phi1 quantiles
        q2 = np.quantile(phi2_arr, [lo, hi], axis=1)  # phi2 quantiles
        lengths = q1[1] - q1[0]
        widths  = q2[1] - q2[0]

    elif method == "kde":
        # -------------------------
        # 1) Determine per-stream min, max, and bin width for phi1
        # -------------------------
        min1 = phi1_arr.min(axis=1)
        max1 = phi1_arr.max(axis=1)
        w1   = (max1 - min1) / nbins  # per-stream bin width

        # -------------------------
        # 2) Digitize phi1 values into bin indices [0, nbins-1]
        # -------------------------
        rel = (phi1_arr - min1[:, None]) / w1[:, None]
        inds = np.floor(rel).astype(int)
        inds = np.clip(inds, 0, nbins - 1)

        # -------------------------
        # 3) Build per-stream histograms via a single bincount
        # -------------------------
        flat = inds + np.arange(S)[:, None] * nbins
        hist = np.bincount(flat.ravel(), minlength=S * nbins).reshape(S, nbins)

        # -------------------------
        # 4) Smooth histogram = approximate KDE
        # -------------------------
        smooth = gaussian_filter1d(hist.astype(float), smoothing_sigma, axis=1)

        # -------------------------
        # 5) Normalize to get a probability density over phi1
        # -------------------------
        area = smooth.sum(axis=1) * w1
        pdf  = smooth / area[:, None]

        # -------------------------
        # 6) Find threshold that encloses density_frac
        # -------------------------
        #   sort bins by density descending
        order = np.argsort(pdf, axis=1)[:, ::-1]
        #   cumulative density in sorted order
        cum   = np.cumsum(np.take_along_axis(pdf, order, axis=1) * w1[:, None], axis=1)
        #   index where cumulative ≥ density_frac
        idx   = np.argmax(cum >= density_frac, axis=1)
        #   threshold density per stream
        thr   = pdf[np.arange(S), order[np.arange(S), idx]]

        # -------------------------
        # 7) Mask bins above threshold to define the coherent phi1 region
        # -------------------------
        mask  = pdf >= thr[:, None]
        #   first and last True per stream
        first = mask.argmax(axis=1)
        last  = nbins - 1 - mask[:, ::-1].argmax(axis=1)

        # -------------------------
        # 8) Compute phi1 length: (last − first) × per-stream bin width
        # -------------------------
        lengths = (last - first) * w1

        # -------------------------
        # 9) Compute phi2 width by quantiles within the phi1 region
        # -------------------------
        lo = (1 - density_frac) / 2
        hi = 1 - lo
        widths = np.empty(S, dtype=float)
        for i in range(S):
            # select particles within the KDE-defined phi1 span
            mask_particles = (
                (phi1_arr[i] >= min1[i] + first[i] * w1[i]) &
                (phi1_arr[i] <= min1[i] + last[i]  * w1[i])
            )
            if mask_particles.sum() < 2:
                widths[i] = 0.0
            else:
                q2 = np.quantile(phi2_arr[i, mask_particles], [lo, hi])
                widths[i] = float(q2[1] - q2[0])
    else:
        raise ValueError("method must be 'quantile' or 'kde'")

    # -------------------------
    # 10) Return scalars if input was 1D
    # -------------------------
    if single:
        return float(lengths[0]), float(widths[0])
    return lengths, widths
