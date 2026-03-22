# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path

import scipy
import scipy.stats
from imageio import imsave
import cv2
import src.utils
import torch
import os
import src.model as m
from tqdm import tqdm
from .evaluation.scores import GridScorer, border_score


def concat_images(images, image_width, spacer_size):
    """Concat image horizontally with spacer"""
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """Concat images in rows"""
    column_size = len(images) // row_size
    spacer_h = (
        np.ones(
            [
                spacer_size,
                image_width * column_size + (column_size - 1) * spacer_size,
                4,
            ],
            dtype=np.uint8,
        )
        * 255
    )

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size * row : column_size * row + column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size - 1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap="jet", smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid="ignore")  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap="jet", smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots // width, activations.shape[-1])
    return rm_fig


def compute_ratemaps(
    model,
    trainer,
    trajectory_generator,
    options,
    res=20,
    n_avg=None,
    Ng=512,
    idxs=None,
):
    """Compute spatial firing fields"""
    if Ng > 64 and options.mode == "train":
        Ng = 64

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    for index in tqdm(range(n_avg)):
        # pos_batch: [batch_size, sequence_length, 2]
        inputs, _, pos_batch = trajectory_generator.get_test_batch()

        if isinstance(model, m.TemporalPCN):
            _, g_batch = trainer.predict(inputs)
            g_batch = (
                g_batch[:, :, :Ng].detach().cpu().numpy().reshape(-1, Ng)
            )  # [sequence_length*batch_size, Ng]
        else:
            g_batch = (
                model.g(inputs)[:, :, :Ng].detach().cpu().numpy().reshape(-1, Ng)
            )  # [sequence_length*batch_size, Ng]

        pos_batch = pos_batch.cpu().detach().numpy().reshape(-1, 2)  # [sequence_length*batch_size, 2]

        g[index] = g_batch
        pos[index] = pos_batch

    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])
    activations = _compute_ratemaps_from_samples(
        xs=pos[:, 0],
        ys=pos[:, 1],
        g=g,
        options=options,
        res=res,
    )

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations


def compute_grid_scores(lo_res, rate_map_lo_res, options, half=False, srted=True):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width = options.box_width
    box_height = options.box_height
    coord_range = ((-box_width / 2, box_width / 2), (-box_height / 2, box_height / 2))
    masks_parameters = zip(starts, ends.tolist())
    scaler = box_height / box_width
    if half:
        scaler = scaler / 2
    lo_res_h = int(lo_res * scaler)
    lo_res_w = lo_res
    scorer = GridScorer(lo_res_w, lo_res_h, coord_range, masks_parameters)

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
        *[scorer.get_scores(rm) for rm in tqdm(rate_map_lo_res)]
    )

    idx = np.flip(np.argsort(score_60))

    if srted:
        return idx, [score_60[i] for i in idx], [sac[i] for i in idx]
    else:
        return idx, score_60, sac


def _compute_ratemaps_from_samples(xs, ys, g, options, res):
    """
    Build occupancy-normalized 2D rate maps from flattened sample arrays.

    Args:
        xs: np.ndarray [T], x positions in meters.
        ys: np.ndarray [T], y positions in meters.
        g: np.ndarray [T, Ng], cell activities.
        options: run options with box_width/box_height.
        res: x-axis resolution.

    Returns:
        np.ndarray [Ng, res_w, res_h] rate maps.
    """
    Ng = g.shape[1]
    scaler = options.box_height / options.box_width
    res_h = int(res * scaler)
    res_w = int(res)
    activations = np.zeros((Ng, res_w, res_h), dtype=float)
    counts = np.zeros((res_w, res_h), dtype=float)

    x_idx = (xs + options.box_width / 2) / options.box_width * res_w
    y_idx = (ys + options.box_height / 2) / options.box_height * res_h

    for i in range(xs.shape[0]):
        x = x_idx[i]
        y = y_idx[i]
        if 0 <= x < res_w and 0 <= y < res_h:
            xi = int(x)
            yi = int(y)
            counts[xi, yi] += 1.0
            activations[:, xi, yi] += g[i, :]

    for x in range(res_w):
        for y in range(res_h):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    return activations


def classify_grid_cells_by_shuffled_null(
    model,
    trainer,
    trajectory_generator,
    options,
    lo_res=20,
    n_avg=None,
    Ng=512,
    n_shuffles=100,
    percentile=95,
    min_shift_steps=1,
    seed=0,
):
    """
    Classify grid cells using a shuffled-null threshold on grid scores.

    This follows the same idea as Stensola-style shuffle controls: keep the
    activity sequence but circularly shift it relative to position samples to
    destroy spatial alignment.

    Args:
        model: Model instance (RNN or TemporalPCN).
        trainer: Corresponding trainer.
        trajectory_generator: Data generator with get_test_batch().
        options: Run options namespace.
        lo_res (int): Ratemap bin resolution on x-axis.
        n_avg (int | None): Number of batches to aggregate. If None, uses
            1000 // sequence_length.
        Ng (int): Number of cells to evaluate (first Ng latent units).
        n_shuffles (int): Number of circular-shift shuffles.
        percentile (float): Null percentile used as significance threshold.
        min_shift_steps (int): Minimum circular shift in sample steps.
        seed (int): RNG seed.

    Returns:
        dict with keys:
            - scores: np.ndarray [Ng], observed grid scores.
            - threshold: float, global null threshold at `percentile`.
            - is_grid: np.ndarray [Ng] bool, scores > threshold.
            - null_scores: np.ndarray [n_shuffles * Ng], shuffled scores.
            - rate_maps: np.ndarray [Ng, lo_res, lo_res_h], observed ratemaps.
    """
    if Ng > options.Ng:
        Ng = options.Ng
    if n_avg is None:
        n_avg = max(1, 1000 // options.sequence_length)

    xs_all = []
    ys_all = []
    g_all = []

    for _ in tqdm(range(n_avg)):
        inputs, _, pos_batch = trajectory_generator.get_test_batch()
        if isinstance(model, m.TemporalPCN):
            _, g_batch = trainer.predict(inputs)
            g_batch = g_batch[:, :, :Ng].detach().cpu().numpy().reshape(-1, Ng)
        else:
            g_batch = model.g(inputs)[:, :, :Ng].detach().cpu().numpy().reshape(-1, Ng)

        pos_batch = pos_batch.detach().cpu().numpy().reshape(-1, 2)
        xs_all.append(pos_batch[:, 0])
        ys_all.append(pos_batch[:, 1])
        g_all.append(g_batch)

    xs = np.concatenate(xs_all, axis=0)
    ys = np.concatenate(ys_all, axis=0)
    g = np.concatenate(g_all, axis=0)  # [T, Ng]
    T = g.shape[0]
    if T < 2:
        raise ValueError("Not enough samples to perform shuffle test.")

    # Observed scores
    rate_maps = _compute_ratemaps_from_samples(xs, ys, g, options, lo_res)
    _, unsrt_scores, _ = compute_grid_scores(lo_res, rate_maps, options, srted=False)
    scores = np.asarray(unsrt_scores, dtype=float)

    # Shuffled null scores (global across cells, as in pooled-null spirit)
    rng = np.random.default_rng(seed)
    max_shift = max(1, T - 1)
    min_shift = int(max(1, min_shift_steps))
    if min_shift >= max_shift:
        min_shift = 1
    null_scores = []
    for _ in tqdm(range(int(n_shuffles))):
        shift = int(rng.integers(min_shift, max_shift + 1))
        g_shift = np.roll(g, shift=shift, axis=0)
        rm_shift = _compute_ratemaps_from_samples(xs, ys, g_shift, options, lo_res)
        _, shuf_scores, _ = compute_grid_scores(lo_res, rm_shift, options, srted=False)
        null_scores.extend(np.asarray(shuf_scores, dtype=float).tolist())

    null_scores = np.asarray(null_scores, dtype=float)
    threshold = float(np.percentile(null_scores, percentile))
    is_grid = scores > threshold

    return {
        "scores": scores,
        "threshold": threshold,
        "is_grid": is_grid,
        "null_scores": null_scores,
        "rate_maps": rate_maps,
    }

def find_global_maxima(field):
    max_index = np.unravel_index(np.argmax(field), field.shape)
    return max_index

def find_local_maxima(autocorr):
    peaks = []
    m, n = autocorr.shape
    for i in range(m):
        for j in range(n):
            val = autocorr[i, j]
            # Check all neighbors in an 8-neighborhood
            is_peak = True
            has_lower_neighbor = False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        neighbor_val = autocorr[ni, nj]
                    else:
                        # Treat out-of-bounds as -inf (so edges can still be peaks if appropriate)
                        neighbor_val = float('-inf')
                    if neighbor_val > val:
                        is_peak = False
                        break
                    if neighbor_val < val:
                        has_lower_neighbor = True
                if not is_peak:
                    break
            # It's a local maximum if no neighbor is higher and at least one neighbor is lower
            if is_peak and has_lower_neighbor:
                peaks.append((i, j))
    
    if not peaks:
        raise RuntimeError("No local maxima found in autocorrelogram.")
    return peaks

def get_grid_scale(peaks, center_peak, pixel_to_meter, method='nearest'):
    nearest_dist = float('inf')
    nearest_peak_coords = None
    cx, cy = center_peak
    dists = []
    for (i, j) in peaks:
        if (i, j) == center_peak:
            continue
        # Euclidean distance between center and this peak
        dist = np.sqrt((i - cx)**2 + (j - cy)**2)
        dists.append(dist)
    nearest_dists = np.sort(dists)[:6]
    grid_scale_px = np.mean(nearest_dists) if method != 'nearest' else nearest_dists[0]
    grid_scale_m = grid_scale_px * pixel_to_meter
    return grid_scale_px, grid_scale_m

def find_central_field_pixels(field, cx, cy, size_thresh_scaler=0.15):
    m, n = field.shape
    center_peak = (cx, cy)
    peak_val = field[cx, cy]
    threshold = size_thresh_scaler * peak_val
    # Flood-fill from the center to get all contiguous cells above threshold
    region_points = []
    visited = np.zeros_like(field, dtype=bool)
    stack = [center_peak]
    visited[cx, cy] = True
    while stack:
        x, y = stack.pop()
        if field[x, y] < threshold:
            continue
        region_points.append((x, y))
        # Explore all neighbors (8-connectivity)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and not visited[nx, ny]:
                    visited[nx, ny] = True
                    stack.append((nx, ny))
    # Ensure the center is included (it should be, given threshold <= peak)
    if center_peak not in region_points:
        region_points.append(center_peak)

    coords = np.array(region_points)
    return coords


def compute_diameter(coords, pixel_to_meter):
    if coords.shape[0] < 2:
        # If region has only one point (unlikely if threshold < peak), diameter is 0
        max_dist = 0.0
    else:
        # Compute pairwise distances and find the maximum
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        max_dist_sq = dist_sq.max()
        max_dist = float(np.sqrt(max_dist_sq))
    grid_size_px = max_dist
    # Convert grid size to meters
    grid_size_m = grid_size_px * pixel_to_meter
    return grid_size_px, grid_size_m


def compute_grid_metrics(autocorr, env_size=1.6, size_thresh_scaler=0.25, scale_method='nearest'):
    """
    Compute grid size and grid scale from a spatial autocorrelogram of a grid cell.
    
    Parameters:
    - autocorr: 2D numpy array (e.g. 39x39) representing the spatial autocorrelogram.
    - env_size: length of one side of the square environment in meters (default 1.6).
    
    Returns:
    A dictionary with:
      'grid_size_pixels': diameter of the center firing field in pixels,
      'grid_size_meters': diameter of the center firing field in meters,
      'grid_scale_pixels': spacing between center and nearest field in pixels,
      'grid_scale_meters': spacing between center and nearest field in meters.
    """
    # Validate input
    if autocorr.ndim != 2:
        raise ValueError("Autocorrelogram must be a 2D array")
    m, n = autocorr.shape
    # Convert grid scale to meters
    pixel_to_meter = env_size / float(m)   # conversion factor per pixel
    
    # 1. Find the local maxima
    peaks = find_local_maxima(autocorr)
    # 2. Find the global maximum
    center_peak = find_global_maxima(autocorr)
    # 3. Compute the grid scale as the distance between the center and the nearest peak
    grid_scale_px, grid_scale_m = get_grid_scale(peaks, center_peak, pixel_to_meter, method=scale_method)
    # 4. Find the central field pixels
    central_field_coords = find_central_field_pixels(autocorr, *center_peak, size_thresh_scaler)
    # 5. Compute the grid size as the diameter of the central field
    grid_size_px, grid_size_m = compute_diameter(central_field_coords, pixel_to_meter)
    
    # 5. Return the results in both pixels and meters, and the nearest peak's coordinates
    return {
        'grid_size_pixels': grid_size_px,
        'grid_size_meters': grid_size_m,
        'grid_scale_pixels': grid_scale_px,
        'grid_scale_meters': grid_scale_m,
    }


def compute_border_scores(lo_res, rate_map_lo_res, options):
    scores = [border_score(rm, lo_res, options.box_width)[0] for rm in rate_map_lo_res]
    idx = np.flip(np.argsort(scores))
    return idx, [scores[i] for i in idx]


# get grid cell rate maps
def compute_1d_ratemaps(
    model,
    trainer,
    trajectory_generator,
    options,
    res=20,
    n_avg=None,
    Ng=512,
    idxs=None,
):
    """Compute spatial firing fields"""

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 1])

    activations = np.zeros([Ng, res])
    counts = np.zeros(res)

    for index in range(n_avg):
        # pos_batch: [batch_size, sequence_length, 2]
        inputs, _, pos_batch = trajectory_generator.get_test_batch()

        # g_batch = model.g(inputs).detach().cpu().numpy().reshape(-1, Ng) # [sequence_length*batch_size, Ng]
        _, g_batch = trainer.predict(inputs)
        g_batch = (
            g_batch.detach().cpu().numpy().reshape(-1, Ng)
        )  # [sequence_length*batch_size, Ng]

        pos_batch = (
            pos_batch.cpu().numpy().reshape(-1, 1)
        )  # [sequence_length*batch_size, 1]

        # g_batch records the activations of all grid cells across all points in all trajectories in this batch
        g_batch = g_batch.reshape(-1, Ng)  # [sequence_length*batch_size, Ng]

        g[index] = g_batch
        pos[index] = pos_batch

        # Convert position (-1, 1) to indices (0, res)
        pos_batch = (
            (pos_batch + options.track_length / 2) / (options.track_length) * res
        )

        for i in range(options.batch_size * options.sequence_length):
            x = pos_batch[i, 0]
            if x >= 0 and x < res:
                counts[int(x)] += 1
                activations[:, int(x)] += g_batch[i, :]

    # make it a density map
    for k in range(res):
        if counts[k] > 0:
            activations[:, k] /= counts[k]

    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 1])
    rate_map = activations.reshape(Ng, -1)

    return rate_map


def save_ratemaps(
    model, trajectory_generator, options, step, res=20, n_avg=None, pc=True
):
    if not n_avg:
        n_avg = 1000 // options.sequence_length
    activations, rate_map, g, pos = compute_ratemaps(
        model, trajectory_generator, options, res=res, n_avg=n_avg, Ng=options.Ng, pc=pc
    )
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    imsave(imdir + "/" + str(step) + ".png", rm_fig)


def save_autocorr(sess, model, save_name, trajectory_generator, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)

    res = dict()
    index_size = 100
    for _ in range(index_size):
        feed_dict = trajectory_generator.feed_dict(flags.box_width, flags.box_height)
        mb_res = sess.run(
            {
                "pos_xy": model.target_pos,
                "bottleneck": model.g,
            },
            feed_dict=feed_dict,
        )
        res = utils.concat_dict(res, mb_res)

    filename = save_name + "/autocorrs_" + str(step) + ".pdf"
    imdir = flags.save_dir + "/"
    out = utils.get_scores_and_plot(
        latest_epoch_scorer, res["pos_xy"], res["bottleneck"], imdir, filename
    )


def plot_1d_performance(place_cell, generator, options, trainer):
    # check if the model generalizes well
    inputs, pc_outputs, pos = generator.get_test_batch()
    pos = pos.cpu()[:5]
    pred_pos = place_cell.get_nearest_cell_pos(trainer.predict(inputs[:5])[0]).cpu()
    centers = place_cell.centers.cpu()
    l = options.track_length / 2

    fig, axes = plt.subplots(5, 1, figsize=(13, 3), sharex=True)

    # Colormap for the time points in each trajectory
    time_cmap = plt.cm.Blues
    test_cmap = plt.cm.Reds

    # Normalize the time points for the colormap
    norm = plt.Normalize(0, pos.shape[1])

    # Plot each trajectory in its own subplot
    for i, (ax, traj) in enumerate(zip(axes, pos)):
        ax.plot(
            centers,
            torch.zeros_like(centers),
            "o",
            color="gray",
            markersize=2,
            alpha=0.2,
        )
        for j, point in enumerate(traj):
            ax.plot(traj[j], 0, "o", color=time_cmap(norm(j)), markersize=5, alpha=0.6)
            ax.plot(pred_pos[i, j], 0, "x", color=test_cmap(norm(j)), markersize=8)

        if i == 0:
            # add a legend
            ax.plot(
                -l,
                0.05,
                "o",
                color=time_cmap(norm(pos.shape[1] - 1)),
                markersize=5,
                alpha=0.6,
            )
            ax.text(-l + l / 20, 0.05, "True Position", va="center", ha="left")
            ax.plot(
                -l, -0.05, "x", color=test_cmap(norm(pos.shape[1] - 1)), markersize=8
            )
            ax.text(-l + l / 20, -0.05, "Predicted Position", va="center", ha="left")

        ax.set_xlim(-l - l / 20, l + l / 20)
        ax.set_ylim(-0.1, 0.1)
        ax.get_yaxis().set_visible(False)  # Hide the y-axis
        # ax.get_xaxis().set_visible(False)  # Hide the x-axis

    fig.text(0.5, 0.01, "Track Position", ha="center", va="center")
    fig.suptitle("Separate Trajectories on a 1D Track")

    # add a colorbar for the time points
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    sm = plt.cm.ScalarMappable(cmap=time_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Time step")

    plt.savefig(os.path.join(options.save_dir, "1d_performance.png"))


def plot_2d_performance(place_cell, generator, options, trainer):

    inputs, pc_outputs, pos = generator.get_test_batch()
    if options.batch_size < 5:
        select = options.batch_size
    else:
        select = 5
    pos = pos.cpu()[:select]
    pred_pos = place_cell.get_nearest_cell_pos(
        trainer.predict(inputs)[0][:select]
    ).cpu()  # size [select, seq_len, 2]
    centers = place_cell.centers.cpu()

    plt.figure(figsize=(5, 5))
    plt.scatter(centers[:, 0], centers[:, 1], s=10, c="k")
    for i in range(select):
        plt.plot(pos[i, :, 0], pos[i, :, 1], c="r", lw=2, label="true")
        plt.plot(pred_pos[i, :, 0], pred_pos[i, :, 1], c="b", lw=2, label="predicted")
        if i == 0:
            plt.legend()

    plt.xlim(-options.box_width / 2, options.box_width / 2)
    plt.ylim(-options.box_height / 2, options.box_height / 2)
    plt.savefig(os.path.join(options.save_dir, "performance.png"))


def plot_1d_ratemaps(rate_map, options):
    n_col = 4
    fig, ax = plt.subplots(
        n_col, options.Ng // n_col, figsize=(2 * options.Ng // n_col, n_col)
    )
    for i, ax in enumerate(ax.flatten()):
        r = (rate_map[i] - rate_map[i].min()) / (rate_map[i].max() - rate_map[i].min())
        ax.plot(
            np.linspace(-options.track_length / 2, options.track_length / 2, 200),
            r,
            lw=3,
        )
        # set overall x and y labels
        if i == 0:
            ax.set_ylabel("Activation")
            ax.set_title(f"Grid Cell {i+1}")
        else:
            ax.get_yaxis().set_visible(False)

        if i == options.Ng // n_col * (n_col - 1):
            ax.set_xlabel("Position")
        else:
            ax.get_xaxis().set_visible(False)

        ax.set_xticks([-options.track_length / 2, 0, options.track_length / 2])
        ax.set_xticklabels([-options.track_length / 2, 0, options.track_length / 2])
    plt.savefig(os.path.join(options.save_dir, "1d_ratemaps.png"))


def trapezoid_mask(h, w, res_h, res_w):
    # Define the trapezoid (vertices in (x, y) format)
    trapezoid_vertices = np.array([[-h/2, -w/2], [h/2, -w/8], [h/2, w/8], [-h/2, w/2]])

    # Create a mask for the trapezoid
    x_indices, y_indices = np.meshgrid(np.linspace(-h/2, h/2, res_h), np.linspace(-w/2, w/2, res_w))  # Pixel coordinates
    points = np.vstack((x_indices.ravel(), y_indices.ravel())).T  # Convert to (N, 2) array

    # Check if each pixel is inside the trapezoid
    trapezoid_path = Path(trapezoid_vertices)
    mask = trapezoid_path.contains_points(points).reshape(res_w, res_h)
    return mask


def plot_2d_ratemaps(rate_map, options, n_col=4, res=30):
    Ng = 64 if options.Ng > 64 else options.Ng
    w = options.box_width
    h = options.box_height
    res_w = int(res)
    res_h = int(res * h / w)

    fig, ax = plt.subplots(n_col, Ng // n_col, figsize=(Ng // n_col, n_col))
    for i, ax in enumerate(ax.flatten()):
        r = (rate_map[i] - rate_map[i].min()) / (
            rate_map[i].max() - rate_map[i].min() + 1e-8
        )
        if options.env_shape == "trapezoid":
            r[~trapezoid_mask(h, w, res_h, res_w)] = np.nan
        ax.imshow(r, cmap="jet", extent=[-h/2, h/2, -w/2, w/2], origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(f'Grid Cell {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(options.save_dir, "2d_ratemaps.png"))


def plot_all_ratemaps(rate_map, options, res=30, scores=None, dir="all_maps"):
    n_col = 8
    Ng_per_file = 64
    n_files = options.Ng // Ng_per_file

    all_dir = os.path.join(options.save_dir, dir)
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)

    w = options.box_width
    h = options.box_height
    res_w = int(res)
    res_h = int(res * h / w)

    for i in tqdm(range(n_files)):
        rm = rate_map[
            i * Ng_per_file : (i + 1) * Ng_per_file
        ]  # [Ng_per_file, res, res]
        fig, axes = plt.subplots(
            Ng_per_file // n_col, n_col, figsize=(n_col, Ng_per_file // n_col)
        )
        for j, ax in enumerate(axes.flatten()):
            r = (rm[j] - rm[j].min()) / (rm[j].max() - rm[j].min() + 1e-8)
            if options.env_shape == "trapezoid":
                r[~trapezoid_mask(h, w, res_h, res_w)] = np.nan
            ax.imshow(r, cmap="jet", extent=[-h/2, h/2, -w/2, w/2], origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if scores:
                ax.set_title(f"{scores[j+i*Ng_per_file]:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(all_dir, f"2d_ratemaps_{i}.png"))
        plt.close(fig)


def plot_top_ratemaps(rate_map, scores, options, n_col=8):
    Ng = len(rate_map)
    fig, ax = plt.subplots(n_col, Ng // n_col, figsize=(Ng // n_col, n_col))
    for i, ax in enumerate(ax.flatten()):
        r = (rate_map[i] - rate_map[i].min()) / (rate_map[i].max() - rate_map[i].min())
        ax.imshow(r, cmap="jet")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{scores[i]:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(options.save_dir, "top_ratemaps.png"))


def plot_loss_err(trainer, options):
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(trainer.err, c="black")
    plt.title("Decoding error (m)")
    plt.xlabel("train step")
    plt.subplot(122)
    plt.plot(trainer.loss, c="black")
    plt.title("Loss")
    plt.xlabel("train step")
    plt.savefig(os.path.join(options.save_dir, "loss"))


def plot_place_cells(place_cell, options, res, n_show=5):
    grid = np.array(
        np.meshgrid(
            np.linspace(-options.box_width / 2, options.box_width / 2, res),
            np.linspace(-options.box_height / 2, options.box_height / 2, res),
        )
    ).T
    grid = torch.tensor(grid, device=options.device)
    grid_pc = place_cell.get_activation(grid)
    select = np.random.choice(np.arange(options.Np), n_show, replace=False)
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2))
    for i in range(n_show):
        # notice the interpolation arg, which visually makes these 'real' place fields
        im = axes[i].imshow(
            grid_pc[:, :, select[i]].cpu().numpy(), cmap="jet", interpolation="bilinear"
        )
        axes[i].set_title("Place cell {}".format(select[i]))
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.6)
    plt.savefig(os.path.join(options.save_dir, "place_cell_examples"))


def plot_same_cells_across_env(
    make_opts_fn,
    get_maps_fn,
    env_sizes=(0.8, 1.2, 1.6, 2.0),
    rf=0.12,
    Np=512,
    cell_ids=(0, 5, 20, 100),
    res=120,
    seed=0,
):
    """
    Plot selected place-cell fields across multiple environment sizes.

    Args:
        make_opts_fn: Callable(L, Np, rf, seed) -> options namespace
        get_maps_fn: Callable(options, res) -> (maps, place_cells)
        env_sizes: Iterable of environment sizes.
        rf: Place-cell receptive field width.
        Np: Number of place cells.
        cell_ids: Sequence of place-cell indices to visualize.
        res: Spatial grid resolution.
        seed: Seed forwarded to make_opts_fn.
    """
    fig, axes = plt.subplots(
        len(cell_ids),
        len(env_sizes),
        figsize=(3 * len(env_sizes), 2.6 * len(cell_ids)),
    )
    if len(cell_ids) == 1:
        axes = np.expand_dims(axes, 0)
    if len(env_sizes) == 1:
        axes = np.expand_dims(axes, 1)

    for c, cell_id in enumerate(cell_ids):
        for e, L in enumerate(env_sizes):
            opts = make_opts_fn(L=L, Np=Np, rf=rf, seed=seed)
            maps, place_cells = get_maps_fn(opts, res=res)

            rm = maps[..., cell_id]
            rm = (rm - rm.min()) / (rm.max() - rm.min() + 1e-8)

            ax = axes[c, e]
            ax.imshow(
                rm,
                origin="lower",
                cmap="jet",
                extent=[-L / 2, L / 2, -L / 2, L / 2],
                vmin=0,
                vmax=1,
            )
            center_x = place_cells.centers[cell_id, 0].cpu()
            center_y = place_cells.centers[cell_id, 1].cpu()
            ax.scatter(
                center_x,
                center_y,
                s=20,
                c="white",
                edgecolors="k",
            )

            if c == 0:
                ax.set_title(f"L={L} m")
            if e == 0:
                ax.set_ylabel(f"Cell {cell_id}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"Same place-cell RF={rf} across environment sizes", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_weights(w, options):
    # plot the weights of the model
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.imshow(w, cmap="jet")
    axes.set_xticks([])
    axes.set_yticks([])
    plt.savefig(os.path.join(options.save_dir, "weights.png"))
