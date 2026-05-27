import numpy as np
from scipy import ndimage
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import json


def extract_sac_fields(sac, corr_thresh=0.1, connectivity=8, min_area=1):
    """
    Extract contiguous SAC fields above a correlation threshold.

    Args:
        sac (np.ndarray):
            Spatial autocorrelogram with shape ``[H, W]``.
        corr_thresh (float):
            Threshold applied to SAC values. Bins with ``sac >= corr_thresh``
            are considered part of candidate fields.
        connectivity (int):
            Neighborhood rule for connected components; one of ``{4, 8}``.
        min_area (int):
            Minimum connected-component area (in bins). Components smaller than
            this are discarded.

    Returns:
        tuple:
          - fields (list[dict]): One dict per surviving field with keys:
            - ``label`` (int): Connected-component label id.
            - ``mask`` (np.ndarray): Bool array of shape ``[H, W]``.
            - ``area`` (int): Number of bins in this field.
            - ``peak_val`` (float): Maximum SAC value inside the field.
            - ``peak_idx`` (tuple[int, int]): ``(row, col)`` index of field peak.
            - ``com`` (tuple[float, float]): Weighted center-of-mass
              ``(row, col)`` using SAC values within the field.
          - labels (np.ndarray): Integer labeled image of shape ``[H, W]``
            where ``0`` is background.
          - binary (np.ndarray): Bool threshold mask of shape ``[H, W]``.

    Method:
        Threshold SAC -> connected-component labeling -> per-component summary
        statistics (peak and COM).
    """
    if sac.ndim != 2:
        raise ValueError("sac must be a 2D array")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    binary = sac >= corr_thresh
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    labels, n = ndimage.label(binary, structure=structure)
    fields = []

    for lab in range(1, n + 1):
        mask = labels == lab
        area = int(mask.sum())
        if area < min_area:
            labels[mask] = 0
            continue

        vals = sac[mask]
        local_idx = int(np.argmax(vals))
        rr, cc = np.where(mask)
        peak_idx = (int(rr[local_idx]), int(cc[local_idx]))

        # Weighted center-of-mass of this field.
        weights = vals.copy()
        wmin = float(weights.min())
        if wmin < 0:
            weights = weights - wmin
        if float(weights.sum()) == 0:
            com = (float(rr.mean()), float(cc.mean()))
        else:
            com = (
                float((rr * weights).sum() / weights.sum()),
                float((cc * weights).sum() / weights.sum()),
            )

        fields.append(
            {
                "label": lab,
                "mask": mask,
                "area": area,
                "peak_val": float(vals.max()),
                "peak_idx": peak_idx,
                "com": com,
            }
        )

    return fields, labels, binary


def grid_scale_stensola_like(fields, sac_shape):
    """
    Estimate grid scale from SAC field centers in a Stensola-like way.

    Args:
        fields (list[dict]):
            Output ``fields`` from :func:`extract_sac_fields`.
            Must contain at least 7 fields (center + 6 surrounding).
        sac_shape (tuple[int, int]):
            Shape of SAC image ``(H, W)`` used to define midpoint.

    Returns:
        dict:
          - ``midpoint`` (tuple[float, float]): SAC midpoint in ``(row, col)``.
          - ``center_field`` (dict): Field nearest the midpoint.
          - ``surrounding_six`` (list[dict]): Six nearest non-center fields.
          - ``distances_px`` (np.ndarray): Shape ``[6]`` distances (pixels) from
            midpoint to surrounding field COMs.
          - ``grid_scale_px`` (float): Mean of ``distances_px``.

    Method:
        Pick center field as COM nearest SAC midpoint, then average distances
        from midpoint to the six closest surrounding COMs.
    """
    if len(fields) < 7:
        raise ValueError("Need at least 7 fields (center + 6 surrounding).")

    midpoint = np.array([(sac_shape[0] - 1) / 2, (sac_shape[1] - 1) / 2], dtype=float)
    coms = np.array([f["com"] for f in fields], dtype=float)

    d_mid = np.linalg.norm(coms - midpoint[None, :], axis=1)
    center_idx = int(np.argmin(d_mid))

    other_idxs = [i for i in range(len(fields)) if i != center_idx]
    d_other = [(i, float(np.linalg.norm(coms[i] - midpoint))) for i in other_idxs]
    d_other.sort(key=lambda x: x[1])

    six_idxs = [i for i, _ in d_other[:6]]
    six_fields = [fields[i] for i in six_idxs]
    six_dists = np.array([d for _, d in d_other[:6]], dtype=float)

    grid_scale_px = float(np.mean(six_dists))
    return {
        "midpoint": tuple(midpoint),
        "center_field": fields[center_idx],
        "surrounding_six": six_fields,
        "distances_px": six_dists,
        "grid_scale_px": grid_scale_px,
    }


def top_cells_grid_scales(
    sacs,
    scores,
    n_cells,
    corr_thresh=0.1,
    connectivity=8,
    min_area=5,
):
    """
    Compute per-cell grid scale for top-scoring cells.

    Args:
        sacs (array-like):
            SAC stack with shape ``[N, H, W]``.
        scores (array-like):
            Score vector with shape ``[N]`` used to rank cells.
        n_cells (int):
            Number of top-scoring cells to evaluate.
        corr_thresh (float):
            Threshold for :func:`extract_sac_fields`.
        connectivity (int):
            Connectivity for :func:`extract_sac_fields` (4 or 8).
        min_area (int):
            Minimum field area in bins.

    Returns:
        np.ndarray:
            Shape ``[min(n_cells, N)]`` of grid scales in pixels.
            Entries are ``np.nan`` where field extraction/scale computation
            fails for a cell.

    Method:
        Sort cells by ``scores`` descending, extract fields per SAC, then call
        :func:`grid_scale_stensola_like`.
    """
    sacs = np.asarray(sacs)
    scores = np.asarray(scores)

    if sacs.ndim != 3:
        raise ValueError("sacs must have shape [N, H, W]")
    if scores.ndim != 1 or len(scores) != len(sacs):
        raise ValueError("scores must be shape [N] and match sacs length")

    n = min(int(n_cells), len(scores))
    top_idx = np.argsort(scores)[::-1][:n]
    scales = np.full(n, np.nan, dtype=float)

    for k, idx in enumerate(top_idx):
        sac = sacs[idx]
        try:
            fields, _, _ = extract_sac_fields(
                sac,
                corr_thresh=corr_thresh,
                connectivity=connectivity,
                min_area=min_area,
            )
            out = grid_scale_stensola_like(fields, sac.shape)
            scales[k] = out["grid_scale_px"]
        except Exception:
            # Keep NaN for cells where extraction/fit fails.
            pass

    return scales


def discreteness_curve_hist(values, bin_widths):
    """
    Histogram-based discreteness score across bin widths.

    Args:
        values (array-like):
            1D sample vector with shape ``[N]``.
        bin_widths (array-like):
            Candidate bin widths with shape ``[B]``.

    Returns:
        np.ndarray:
            Shape ``[B]`` discreteness score per bin width.
            Score is ``std(hist_counts)``.

    Method:
        For each bin width, build histogram on fixed range and compute the
        standard deviation of bin counts. This follows the discreteness
        definition described in Stensola supplementary methods.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("Need at least 3 finite values.")

    x_min, x_max = float(np.min(x)), float(np.max(x))
    out = []
    for bw in bin_widths:
        bw = float(bw)
        if bw <= 0:
            out.append(np.nan)
            continue
        n_bins = max(3, int(np.ceil((x_max - x_min) / bw)))
        counts, _ = np.histogram(x, bins=n_bins, range=(x_min, x_max))
        if counts.size < 2:
            out.append(np.nan)
            continue
        out.append(float(np.std(counts)))
    return np.asarray(out, dtype=float)


def continuous_null_curve(values, bin_widths, n_shuffle=100, random_state=0):
    """
    Generate a jitter-shuffled null baseline for the discreteness curve.

    Args:
        values (array-like):
            1D sample vector with shape ``[N]``.
        bin_widths (array-like):
            Bin-width vector with shape ``[B]``.
        n_shuffle (int):
            Number of null shuffles (paper used 100).
        random_state (int):
            RNG seed.

    Returns:
        tuple[np.ndarray, np.ndarray]:
          - null_mean: shape ``[B]`` mean discreteness over resamples.
          - null_std: shape ``[B]`` std discreteness over resamples.

    Method:
        For each shuffle, add independent uniform jitter to every spacing:
        ``x_i + U(-0.5 * min(x), +0.5 * min(x))``. This preserves overall
        distribution shape while removing local discontinuities, as described
        in the Stensola supplementary methods.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("Need at least 3 finite values.")
    rng = np.random.default_rng(random_state)
    min_spacing = float(np.min(x))
    jitter_amp = 0.5 * min_spacing
    curves = []
    for _ in range(int(n_shuffle)):
        jitter = rng.uniform(-jitter_amp, jitter_amp, size=x.size)
        sample = x + jitter
        # Grid spacing should remain positive.
        sample = np.maximum(sample, 1e-12)
        curves.append(discreteness_curve_hist(sample, bin_widths))
    curves = np.asarray(curves, dtype=float)
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)


def estimate_k_from_ksd(
    log_scales,
    k_max=6,
    prominence=0.05,
    grid_size=512,
    bw_mode="scott",
    exp_bw_a=0.8,
    exp_bw_b=0.01,
    exp_bw_min=0.02,
    exp_bw_max=1.0,
):
    """
    Estimate module count ``k`` from KDE peaks on log-scales.

    Args:
        log_scales (array-like):
            1D vector with shape ``[N]`` containing ``log(scale)`` values.
        k_max (int):
            Upper cap for returned ``k``.
        prominence (float):
            Relative peak-prominence threshold in ``[0, 1+]`` applied to KDE
            dynamic range.
        grid_size (int):
            Number of points in evaluation grid for KDE.
        bw_mode (str):
            KDE bandwidth mode: ``"scott"`` (default), ``"silverman"``,
            or ``"exp"`` for an explicit sample-size exponential schedule.
        exp_bw_a (float):
            Exponential bandwidth scale ``a`` used when ``bw_mode="exp"``.
        exp_bw_b (float):
            Exponential decay rate ``b`` used when ``bw_mode="exp"``.
        exp_bw_min (float):
            Lower clamp for exponential bandwidth factor.
        exp_bw_max (float):
            Upper clamp for exponential bandwidth factor.

    Returns:
        tuple:
          - k (int): Estimated number of modules.
          - ksd (dict | None):
            ``None`` for very small samples; otherwise
            ``{'grid': [G], 'density': [G], 'peaks': [P]}``.

    Method:
        Fit 1D KDE on ``log_scales``, detect prominent density peaks, and map
        peak count to a bounded ``k``.
    """
    x = np.asarray(log_scales, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return 1, None
    n = int(x.size)
    bw_mode = str(bw_mode).lower()
    if bw_mode == "scott":
        kde = gaussian_kde(x, bw_method="scott")
    elif bw_mode == "silverman":
        kde = gaussian_kde(x, bw_method="silverman")
    elif bw_mode == "exp":
        # Explicit sample-size schedule: bw = a * exp(-b * n), clamped.
        bw = float(exp_bw_a) * np.exp(-float(exp_bw_b) * float(n))
        bw = float(np.clip(bw, float(exp_bw_min), float(exp_bw_max)))
        kde = gaussian_kde(x, bw_method=bw)
    else:
        raise ValueError("bw_mode must be one of {'scott', 'silverman', 'exp'}")
    grid = np.linspace(np.min(x), np.max(x), int(grid_size))
    dens = kde(grid)
    prom_abs = float(prominence) * (np.max(dens) - np.min(dens) + 1e-12)
    peaks, _ = find_peaks(dens, prominence=prom_abs)
    k = int(np.clip(max(1, len(peaks)), 1, int(k_max)))
    return k, {"grid": grid, "density": dens, "peaks": peaks}


def _build_feature_matrix(scales, orientations=None, orientation_weight=1.0):
    """
    Build clustering feature matrix from scale and optional orientation.

    Args:
        scales (array-like):
            1D scales with shape ``[N]``.
        orientations (array-like | None):
            Optional orientation angles in degrees, shape ``[N]``.
        orientation_weight (float):
            Multiplicative weight applied to orientation features.

    Returns:
        np.ndarray:
          - Shape ``[N, 1]`` if orientations is ``None`` (``log(scale)`` only).
          - Shape ``[N, 3]`` if orientations provided:
            ``[log(scale), w*cos(theta), w*sin(theta)]``.
    """
    x = np.asarray(scales, dtype=float)
    lx = np.log(x)
    if orientations is None:
        return lx[:, None]
    th = np.asarray(orientations, dtype=float)
    if th.shape[0] != lx.shape[0]:
        raise ValueError("orientations must match scales length")
    return np.column_stack(
        [lx, orientation_weight * np.cos(np.deg2rad(th)), orientation_weight * np.sin(np.deg2rad(th))]
    )


def _within_between_ratio(log_scales, labels):
    """
    Compute within-cluster vs between-cluster pairwise distance summary.

    Args:
        log_scales (array-like):
            1D vector with shape ``[N]``.
        labels (array-like):
            Cluster labels with shape ``[N]``.

    Returns:
        tuple[float, float, float]:
          - within_mean: mean pairwise |delta| for same-label pairs.
          - between_mean: mean pairwise |delta| for different-label pairs.
          - between_over_within: ratio ``between_mean / within_mean``.
    """
    x = np.asarray(log_scales, dtype=float)
    y = np.asarray(labels)
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan
    within = []
    between = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(x[i] - x[j])
            if y[i] == y[j]:
                within.append(d)
            else:
                between.append(d)
    within_m = float(np.mean(within)) if within else np.nan
    between_m = float(np.mean(between)) if between else np.nan
    ratio = float(between_m / (within_m + 1e-12)) if np.isfinite(within_m) and np.isfinite(between_m) else np.nan
    return within_m, between_m, ratio


def module_discreteness_report(
    scales,
    orientations=None,
    k_max=6,
    n_kmeans_runs=300,
    min_cluster_size=3,
    bin_widths=None,
    n_shuffle=100,
    random_state=0,
    orientation_weight=1.0,
    ksd_bw_mode="scott",
    ksd_exp_bw_a=0.8,
    ksd_exp_bw_b=0.01,
    ksd_exp_bw_min=0.02,
    ksd_exp_bw_max=1.0,
):
    """
    Quantitative discreteness report inspired by Stensola supplementary analyses.

    Args:
        scales (array-like):
            Per-cell scale values (e.g., cm or px), shape ``[N]``.
            Must be positive and finite.
        orientations (array-like | None):
            Optional per-cell orientation angles (degrees), shape ``[N]``.
            If provided, orientation features are included in clustering.
        k_max (int):
            Maximum allowed module count.
        n_kmeans_runs (int):
            Number of random k-means restarts; best silhouette is selected.
        min_cluster_size (int):
            Clusters smaller than this are excluded from module summaries.
        bin_widths (array-like | None):
            Optional bin-width vector shape ``[B]`` for discreteness curve.
            If None, an automatic range is generated.
        n_shuffle (int):
            Number of jitter-null shuffles (paper used 100).
        random_state (int):
            RNG seed for reproducibility.
        orientation_weight (float):
            Relative weight for orientation features in clustering.
        ksd_bw_mode (str):
            Bandwidth mode forwarded to :func:`estimate_k_from_ksd`.
            One of ``{"scott", "silverman", "exp"}``.
        ksd_exp_bw_a (float):
            Exponential bandwidth scale ``a`` when ``ksd_bw_mode="exp"``.
        ksd_exp_bw_b (float):
            Exponential bandwidth decay ``b`` when ``ksd_bw_mode="exp"``.
        ksd_exp_bw_min (float):
            Lower clamp for exponential bandwidth factor.
        ksd_exp_bw_max (float):
            Upper clamp for exponential bandwidth factor.

    Returns:
        dict with keys:
          - ``n_cells_used`` (int)
          - ``scales_used`` (np.ndarray, ``[N_valid]``)
          - ``log_scales_used`` (np.ndarray, ``[N_valid]``)
          - ``orientations_used`` (np.ndarray | None, ``[N_valid]``)
          - ``bin_widths`` (np.ndarray, ``[B]``)
          - ``discreteness_curve`` (np.ndarray, ``[B]``)
          - ``null_curve_mean`` (np.ndarray, ``[B]``)
          - ``null_curve_std`` (np.ndarray, ``[B]``)
          - ``discreteness_ratio_curve`` (np.ndarray, ``[B]``)
          - ``discreteness_ratio_peak`` (float)
          - ``k_estimate_ksd`` (int)
          - ``ksd`` (dict | None): KDE grid/density/peak indices
          - ``k_used_kmeans`` (int)
          - ``labels`` (np.ndarray, ``[N_valid]``)
          - ``silhouette_best`` (float)
          - ``cluster_sizes`` (dict[int, int])
          - ``module_scale_means`` (dict[int, float])
          - ``adjacent_scale_ratios`` (list[float])
          - ``within_between_logscale`` (dict):
            ``within_mean``, ``between_mean``, ``between_over_within``

    Method:
        1) Filter valid scales/orientations.
        2) Compute histogram discreteness curve and jitter-shuffled null baseline.
        3) Estimate ``k`` via KDE peak count on ``log(scales)``.
        4) Run repeated k-means and pick solution with highest silhouette.
        5) Build module summaries and within/between separation statistics.
    """
    scales = np.asarray(scales, dtype=float)
    mask = np.isfinite(scales) & (scales > 0)
    if orientations is not None:
        orientations = np.asarray(orientations, dtype=float)
        mask &= np.isfinite(orientations)
        orientations = orientations[mask]
    scales = scales[mask]
    if scales.size < 6:
        raise ValueError("Need at least 6 valid scales.")

    if bin_widths is None:
        x_min, x_max = float(np.min(scales)), float(np.max(scales))
        hi = max((x_max - x_min) / 6.0, 1e-6)
        lo = 3.0 if hi >= 3.0 else max(hi / 10.0, 1e-6)
        bin_widths = np.linspace(lo, hi, 15)
    bin_widths = np.asarray(bin_widths, dtype=float)

    disc_curve = discreteness_curve_hist(scales, bin_widths)
    null_mean, null_std = continuous_null_curve(scales, bin_widths, n_shuffle=n_shuffle, random_state=random_state)
    disc_ratio_curve = disc_curve / (null_mean + 1e-12)
    disc_ratio_peak = float(np.nanmax(disc_ratio_curve))

    log_scales = np.log(scales)
    k_est, ksd = estimate_k_from_ksd(
        log_scales,
        k_max=k_max,
        bw_mode=ksd_bw_mode,
        exp_bw_a=ksd_exp_bw_a,
        exp_bw_b=ksd_exp_bw_b,
        exp_bw_min=ksd_exp_bw_min,
        exp_bw_max=ksd_exp_bw_max,
    )

    X = _build_feature_matrix(scales, orientations=orientations, orientation_weight=orientation_weight)
    k = int(np.clip(k_est, 1, min(int(k_max), X.shape[0] - 1)))
    if k < 2:
        labels = np.zeros(X.shape[0], dtype=int)
        sil_best = np.nan
    else:
        best = {"sil": -np.inf, "labels": None}
        rng = np.random.default_rng(random_state)
        for _ in range(int(n_kmeans_runs)):
            seed = int(rng.integers(0, 2**31 - 1))
            km = KMeans(n_clusters=k, n_init=1, random_state=seed)
            labels = km.fit_predict(X)
            # silhouette requires at least 2 clusters and non-singleton labels pattern
            if len(np.unique(labels)) < 2:
                continue
            try:
                sil = float(silhouette_score(X, labels))
            except Exception:
                continue
            if sil > best["sil"]:
                best = {"sil": sil, "labels": labels}
        labels = best["labels"] if best["labels"] is not None else np.zeros(X.shape[0], dtype=int)
        sil_best = best["sil"] if best["labels"] is not None else np.nan

    # Remove tiny clusters from module summaries (Stensola-style outlier control)
    uniq, cnt = np.unique(labels, return_counts=True)
    keep_clusters = set(uniq[cnt >= int(min_cluster_size)])
    keep_mask = np.array([lab in keep_clusters for lab in labels], dtype=bool)
    labels_kept = labels[keep_mask]
    scales_kept = scales[keep_mask]
    log_kept = np.log(scales_kept)

    cluster_sizes = {int(u): int(c) for u, c in zip(uniq, cnt)}
    module_scale_means = {}
    for lab in sorted(keep_clusters):
        module_scale_means[int(lab)] = float(np.mean(scales[labels == lab]))
    means_sorted = np.array(sorted(module_scale_means.values()), dtype=float)
    adj_ratios = (means_sorted[1:] / means_sorted[:-1]).tolist() if means_sorted.size >= 2 else []

    within_m, between_m, wb_ratio = _within_between_ratio(log_kept, labels_kept) if scales_kept.size >= 2 else (np.nan, np.nan, np.nan)

    return {
        "n_cells_used": int(scales.size),
        "scales_used": scales,
        "log_scales_used": log_scales,
        "orientations_used": orientations,
        "bin_widths": bin_widths,
        "discreteness_curve": disc_curve,
        "null_curve_mean": null_mean,
        "null_curve_std": null_std,
        "discreteness_ratio_curve": disc_ratio_curve,
        "discreteness_ratio_peak": disc_ratio_peak,
        "k_estimate_ksd": int(k_est),
        "ksd": ksd,
        "k_used_kmeans": int(k),
        "labels": labels,
        "silhouette_best": sil_best,
        "cluster_sizes": cluster_sizes,
        "module_scale_means": module_scale_means,
        "adjacent_scale_ratios": adj_ratios,
        "within_between_logscale": {
            "within_mean": within_m,
            "between_mean": between_m,
            "between_over_within": wb_ratio,
        },
    }


def plot_module_discreteness_report(report, bins=24, figsize=(15, 4)):
    """
    Plot a compact summary of a module_discreteness_report.

    Args:
        report (dict):
            Output from :func:`module_discreteness_report`.
        bins (int):
            Histogram bins for scale-distribution panel.
        figsize (tuple[float, float]):
            Matplotlib figure size.

    Returns:
        tuple:
          - fig (matplotlib.figure.Figure)
          - axes (np.ndarray): shape ``[3]`` axis array

    Panels:
        1) Discreteness curve vs null mean ± 1 std.
        2) KDE(log-scale) with detected peaks and summary title.
        3) Scale histogram with vertical lines at module mean scales and
           adjacent ratio text.
    """
    import matplotlib.pyplot as plt

    bw = np.asarray(report["bin_widths"], dtype=float)
    disc = np.asarray(report["discreteness_curve"], dtype=float)
    null_m = np.asarray(report["null_curve_mean"], dtype=float)
    null_s = np.asarray(report["null_curve_std"], dtype=float)
    ratio = np.asarray(report["discreteness_ratio_curve"], dtype=float)
    scales = np.asarray(report["scales_used"], dtype=float)
    labels = np.asarray(report["labels"])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: discreteness vs null baseline
    ax = axes[0]
    ax.plot(bw, disc, lw=2, label="data discreteness")
    ax.plot(bw, null_m, lw=2, label="jitter null mean")
    ax.fill_between(bw, null_m - null_s, null_m + null_s, alpha=0.2, label="null ±1 sd")
    ax.set_xlabel("Bin width")
    ax.set_ylabel("Discreteness score")
    ax.set_title(f"Peak ratio={report['discreteness_ratio_peak']:.2f}")
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: KSD (if available)
    ax = axes[1]
    ksd = report.get("ksd", None)
    if ksd is not None:
        grid = np.asarray(ksd["grid"], dtype=float)
        dens = np.asarray(ksd["density"], dtype=float)
        peaks = np.asarray(ksd["peaks"], dtype=int)
        ax.plot(grid, dens, lw=2, label="KSD(log scale)")
        if peaks.size > 0:
            ax.scatter(grid[peaks], dens[peaks], s=30, zorder=3, label="peaks")
        ax.set_xlabel("log(scale)")
        ax.set_ylabel("Density")
    else:
        ax.text(0.5, 0.5, "KSD unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f"k_est={report['k_estimate_ksd']}, sil={report['silhouette_best']:.2f}")
    ax.legend(frameon=False, fontsize=8, loc="best")

    # Panel 3: scale distribution + module means
    ax = axes[2]
    ax.hist(scales, bins=bins, color="lightgray", edgecolor="white")
    means = report.get("module_scale_means", {})
    for i, m in enumerate(sorted(means.values())):
        ax.axvline(m, ls="--", lw=1.8, label="module means" if i == 0 else None)
    ratios = report.get("adjacent_scale_ratios", [])
    ratio_txt = ", ".join(f"{r:.2f}" for r in ratios) if len(ratios) else "n/a"
    ax.set_title(f"Adjacent ratios: {ratio_txt}")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Count")
    if len(means) > 0:
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig, axes


def _module_counts_from_cfg(ng, n_module, module_fracs):
    """Return per-module counts from config fractions (same logic as notebooks)."""
    ng = int(ng)
    n_module = int(n_module)
    if n_module < 1:
        raise ValueError("n_module must be >= 1.")
    if n_module == 1:
        return [ng]

    if module_fracs is None:
        if ng % n_module != 0:
            raise ValueError(
                f"Ng={ng} not divisible by n_module={n_module} and module_fracs missing."
            )
        return [ng // n_module] * n_module

    fr = np.asarray(module_fracs, dtype=float).flatten()
    if fr.size != n_module:
        raise ValueError(
            f"len(module_fracs)={fr.size} does not match n_module={n_module}."
        )
    if np.any(fr < 0):
        raise ValueError("module_fracs must be non-negative.")
    if np.allclose(fr.sum(), 0):
        raise ValueError("module_fracs sum must be > 0.")

    fr = fr / fr.sum()
    counts = np.floor(fr * ng).astype(int)
    counts[-1] = ng - counts[:-1].sum()
    return counts.tolist()


def module_index_by_sorted_score(run_path, n_cells=None):
    """
    Return module index aligned to score-sorted order.

    Args:
        run_path (str): Path to run directory containing configs/scores.
        n_cells (int | None): Optional truncation in sorted order.

    Returns:
        tuple[np.ndarray, dict]:
          - module_idx_sorted (np.ndarray): shape [Ng] or [n_cells]
          - cfg (dict): loaded config dict
    """
    with open(os.path.join(run_path, "configs.json"), "r") as f:
        cfg = json.load(f)

    n_module = int(cfg.get("n_module", 1))
    counts = _module_counts_from_cfg(
        ng=int(cfg["Ng"]),
        n_module=n_module,
        module_fracs=cfg.get("module_fracs", None),
    )

    unsrt_scores = np.asarray(np.load(os.path.join(run_path, "unsrt_grid_scores.npy"))).reshape(-1)
    sort_idx = np.argsort(unsrt_scores)[::-1]
    ng = len(unsrt_scores)

    module_of_orig = np.empty(ng, dtype=int)
    s = 0
    for m, c in enumerate(counts):
        module_of_orig[s:s + c] = m
        s += c
    module_idx_sorted = module_of_orig[sort_idx]
    if n_cells is not None:
        module_idx_sorted = module_idx_sorted[: int(n_cells)]
    return module_idx_sorted, cfg


def get_grid_scales_cm(
    dt,
    model="tpc",
    n_cells=80,
    h=None,
):
    """
    Canonical grid-scale extraction (notebook 'me' method).

    This function computes per-cell scale from SAC local maxima and center peak.
    """
    # Local import to avoid heavy module coupling unless needed.
    from src.visualize import find_local_maxima, find_global_maxima, get_grid_scale

    path = f"../results/{model}/{dt}"
    if h is None:
        with open(os.path.join(path, "configs.json"), "r") as f:
            cfg = json.load(f)
        h = float(cfg.get("box_height", 1.6))
    sacs = np.load(f"{path}/sac.npy")[:n_cells]  # score-sorted
    scores = np.load(f"{path}/grid_scores.npy")
    unsrt_scores = np.load(f"{path}/unsrt_grid_scores.npy")

    scales_px = np.array(
        [
            get_grid_scale(
                find_local_maxima(sac),
                find_global_maxima(sac),
                method="average",
            )
            for sac in sacs
        ],
        dtype=float,
    )

    rate_map_h = (sacs[0].shape[0] + 1) / 2
    bin_size_m = h / rate_map_h
    scales_cm = scales_px * bin_size_m * 100.0
    return scales_cm, scales_px, scores[: len(sacs)], unsrt_scores, sacs


def get_grid_scales_cm_stensola(
    dt,
    model="tpc",
    n_cells=80,
    corr_thresh=0.2,
    h=None,
    connectivity=8,
    min_area=5,
):
    """
    Stensola-like scale extraction kept as a separate method/cache path.
    """
    path = f"../results/{model}/{dt}"
    if h is None:
        with open(os.path.join(path, "configs.json"), "r") as f:
            cfg = json.load(f)
        h = float(cfg.get("box_height", 1.6))
    sacs = np.load(f"{path}/sac.npy")  # full stack
    scores = np.load(f"{path}/grid_scores.npy")
    unsrt_scores = np.load(f"{path}/unsrt_grid_scores.npy")

    scales_px = top_cells_grid_scales(
        sacs,
        scores,
        n_cells=n_cells,
        corr_thresh=corr_thresh,
        connectivity=connectivity,
        min_area=min_area,
    )

    rate_map_h = (sacs[0].shape[0] + 1) / 2
    bin_size_m = h / rate_map_h
    scales_cm = scales_px * bin_size_m * 100.0
    return scales_cm, scales_px, scores[: min(int(n_cells), len(scores))], unsrt_scores, sacs[: min(int(n_cells), len(sacs))]


def _wrap_deg(x):
    """Wrap angle in degrees to [-180, 180)."""
    return (x + 180.0) % 360.0 - 180.0


def _label_axes_stensola(center_peak, local_peaks, top_k=6):
    """Notebook-equivalent Stensola axis labeling from nearest ring peaks."""
    c = np.asarray(center_peak, dtype=float)
    ring = [np.asarray(p, dtype=float) for p in local_peaks if not np.array_equal(p, c)]
    if len(ring) < 3:
        return None

    ring = sorted(ring, key=lambda p: np.linalg.norm(p - c))[:top_k]
    vecs = [p - c for p in ring]  # (dr, dc)

    # horizontal reference = 0 deg (to the right); use -dr for math-style y-up
    ang = np.array([_wrap_deg(np.degrees(np.arctan2(-v[0], v[1]))) for v in vecs], dtype=float)

    i1 = int(np.argmin(np.abs(ang)))  # Axis 1: closest to horizontal
    a1 = ang[i1]
    d = np.array([_wrap_deg(a - a1) for a in ang], dtype=float)
    d[i1] = np.nan

    pos = np.where(d > 0)[0]
    neg = np.where(d < 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None

    i2 = int(pos[np.argmin(np.abs(d[pos]))])  # Axis 2: closest positive
    i3 = int(neg[np.argmin(np.abs(d[neg]))])  # Axis 3: closest negative
    return ang[i1], ang[i2], ang[i3]


def _circular_mean_60(angles_deg):
    """Notebook-equivalent 60-periodic circular mean."""
    a = np.asarray(angles_deg, dtype=float)
    z = np.exp(1j * 2.0 * np.pi * a / 60.0)
    m = np.mean(z)
    if np.abs(m) < 1e-10:
        return np.nan
    return (np.degrees(np.angle(m)) * 60.0 / 360.0)


def get_grid_orientations_deg(dt, model="tpc", n_cells=80, top_k=6):
    """
    Stensola-style orientation extraction from SACs.

    Returns orientations and per-cell axis triplets aligned to score-sorted cells.
    """
    from src.visualize import find_local_maxima, find_global_maxima

    path = f"../results/{model}/{dt}"
    sacs = np.load(f"{path}/sac.npy")[:n_cells]  # score-sorted
    scores = np.load(f"{path}/grid_scores.npy")
    unsrt_scores = np.load(f"{path}/unsrt_grid_scores.npy")

    orientations_deg = []
    axes_deg = []
    for sac in sacs:
        center = find_global_maxima(sac)
        peaks = find_local_maxima(sac)
        ax = _label_axes_stensola(center, peaks, top_k=top_k)
        if ax is None:
            orientations_deg.append(np.nan)
            axes_deg.append((np.nan, np.nan, np.nan))
            continue
        a1, a2, a3 = ax
        orientations_deg.append(_circular_mean_60([a1, a2, a3]))
        axes_deg.append((a1, a2, a3))

    return (
        np.asarray(orientations_deg, dtype=float),
        np.asarray(axes_deg, dtype=float),
        scores[: len(sacs)],
        unsrt_scores,
        sacs,
    )


def summarize_by_module(values, module_idx_sorted):
    """
    Compute per-module mean/std/count for a 1D metric aligned to sorted cells.
    """
    v = np.asarray(values, dtype=float)
    m = np.asarray(module_idx_sorted, dtype=int)
    mask = np.isfinite(v)
    v = v[mask]
    m = m[mask]
    out = {}
    for mod in sorted(np.unique(m)):
        g = v[m == mod]
        out[int(mod)] = {
            "mean": float(np.mean(g)),
            "std": float(np.std(g, ddof=1)) if g.size > 1 else np.nan,
            "n": int(g.size),
        }
    return out


def _select_cells_by_scale(scales_cm, n_examples=6, selection="first"):
    """
    Return cell indices selected from scale values.

    selection:
      - "first": first n cells in current order
      - "largest" / "highest": n cells with largest finite scales
      - "smallest" / "lowest": n cells with smallest finite scales
    """
    scales = np.asarray(scales_cm, dtype=float)
    n = min(int(n_examples), len(scales))
    selection = str(selection).lower()
    if selection == "first":
        return np.arange(n)

    valid = np.where(np.isfinite(scales) & (scales > 0))[0]
    if selection in ("largest", "highest", "max"):
        order = valid[np.argsort(scales[valid])[::-1]]
    elif selection in ("smallest", "lowest", "min"):
        order = valid[np.argsort(scales[valid])]
    else:
        raise ValueError(
            "selection must be one of {'first', 'largest', 'highest', 'smallest', 'lowest'}"
        )
    return order[:n]


def _nearest_ring_peaks(center_peak, local_peaks, n_peaks=6):
    """Return nearest local peaks excluding the central SAC peak."""
    c = np.asarray(center_peak, dtype=float)
    ring = [
        np.asarray(p, dtype=float)
        for p in local_peaks
        if not np.array_equal(np.asarray(p), c)
    ]
    ring = sorted(ring, key=lambda p: np.linalg.norm(p - c))
    return ring[: int(n_peaks)]


def plot_sac_examples_with_scale_orientation(
    sacs,
    scales_cm,
    orientations_deg,
    scores=None,
    n_examples=6,
    selection="first",
    indices=None,
    draw_peak_connections=False,
    n_peaks=6,
    cmap="viridis",
    connection_color="white",
):
    """
    Plot SAC examples with scale and orientation annotations.

    Args:
        sacs: SAC stack aligned to ``scales_cm``.
        scales_cm: Per-cell grid scales.
        orientations_deg: Per-cell orientations.
        scores: Optional per-cell grid scores.
        n_examples: Number of cells to plot.
        selection: "first", "largest"/"highest", or "smallest"/"lowest".
        indices: Optional explicit cell indices to plot. Takes precedence over
            ``selection`` and ``n_examples``.
        draw_peak_connections: If True, draw lines from SAC center to the
            nearest ring peaks.
        n_peaks: Number of ring peaks to connect when drawing overlays.
        cmap: Matplotlib colormap for SACs.
        connection_color: Color for center/peak connection overlays.
    """
    import matplotlib.pyplot as plt
    from src.visualize import find_global_maxima, find_local_maxima

    if indices is None:
        idx = _select_cells_by_scale(scales_cm, n_examples=n_examples, selection=selection)
    else:
        idx = np.asarray(indices, dtype=int)
    n = len(idx)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    axes = axes[0]

    for ax, i in zip(axes, idx):
        sac = sacs[i]
        ax.imshow(sac, cmap=cmap, interpolation="gaussian")
        if draw_peak_connections:
            center = np.asarray(find_global_maxima(sac), dtype=float)
            try:
                peaks = _nearest_ring_peaks(
                    center,
                    find_local_maxima(sac),
                    n_peaks=n_peaks,
                )
                ax.scatter(center[1], center[0], s=24, c="white", edgecolors="black")
                for peak in peaks:
                    ax.plot(
                        [center[1], peak[1]],
                        [center[0], peak[0]],
                        color=connection_color,
                        lw=1.4,
                        alpha=0.9,
                    )
                    ax.scatter(peak[1], peak[0], s=18, c=connection_color, edgecolors="none")
            except RuntimeError:
                pass
        title = f"idx {i}\nscale={scales_cm[i]:.2f} cm\nori={orientations_deg[i]:.2f} deg"
        if scores is not None and i < len(scores):
            title += f"\nscore={scores[i]:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, axes


def plot_grid_map_examples_with_scale_orientation(
    grid_maps,
    scales_cm,
    orientations_deg=None,
    scores=None,
    n_examples=6,
    selection="first",
    cmap="jet",
):
    """
    Plot grid-map examples with the same scale-based selection as SAC examples.

    ``grid_maps`` should be aligned to ``scales_cm`` and typically comes from
    ``results/<model>/<run>/grid_maps.npy``.
    """
    import matplotlib.pyplot as plt

    idx = _select_cells_by_scale(scales_cm, n_examples=n_examples, selection=selection)
    n = len(idx)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    axes = axes[0]

    for ax, i in zip(axes, idx):
        rm = np.asarray(grid_maps[i], dtype=float)
        rm = (rm - np.nanmin(rm)) / (np.nanmax(rm) - np.nanmin(rm) + 1e-8)
        ax.imshow(rm, cmap=cmap, origin="lower", interpolation="gaussian")

        title = f"idx {i}\nscale={scales_cm[i]:.2f} cm"
        if orientations_deg is not None and i < len(orientations_deg):
            title += f"\nori={orientations_deg[i]:.2f} deg"
        if scores is not None and i < len(scores):
            title += f"\nscore={scores[i]:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, axes
