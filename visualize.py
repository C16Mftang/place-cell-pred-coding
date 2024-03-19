# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
import cv2
import utils
import torch
import os

def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
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
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def compute_ratemaps(
        model, 
        trajectory_generator,  
        options, 
        res=20, 
        n_avg=None, 
        Ng=512, 
        idxs=None,
        pc=False,
    ):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        # pos_batch: [batch_size, seq_len, 2]
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()

        if pc:
            # when model is tpc, we need a initial state
            init_state = inputs[1] @ trajectory_generator.get_hidden_projector()
            g_batch = model.g(inputs[0], init_state).detach().cpu().numpy() # [seq_len, batch_size, Ng]
        else:
            g_batch = model.g(inputs).detach().cpu().numpy() # [seq_len, batch_size, Ng]
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])

        # g_batch records the activations of all grid cells across all points in all trajectories in this batch
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng) # [seq_len*batch_size, Ng]
        
        g[index] = g_batch
        pos[index] = pos_batch

        # Convert position to indices
        # add h/2 or w/2 is to transform the top-left corner to the center of the box
        # divide by h or w is to normalize the position to [0, 1] to fit the resolution
        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    # make it a density map
    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos


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
    '''Compute spatial firing fields'''

    g = np.zeros([n_avg, options.batch_size * options.seq_len, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.seq_len, 1])

    activations = np.zeros([Ng, res]) 
    counts  = np.zeros(res)

    for index in range(n_avg):
        # pos_batch: [batch_size, seq_len, 2]
        inputs, _, pos_batch = trajectory_generator.get_test_batch()

        # g_batch = model.g(inputs).detach().cpu().numpy().reshape(-1, Ng) # [seq_len*batch_size, Ng]
        _, g_batch = trainer.predict(inputs)
        g_batch = g_batch.detach().cpu().numpy().reshape(-1, Ng) # [seq_len*batch_size, Ng]
        
        pos_batch = pos_batch.cpu().numpy().reshape(-1, 1) # [seq_len*batch_size, 1]

        # g_batch records the activations of all grid cells across all points in all trajectories in this batch
        g_batch = g_batch.reshape(-1, Ng) # [seq_len*batch_size, Ng]
        
        g[index] = g_batch
        pos[index] = pos_batch

        # Convert position (-1, 1) to indices (0, res)
        pos_batch = (pos_batch + options.track_length/2) / (options.track_length) * res

        for i in range(options.batch_size*options.seq_len):
            x = pos_batch[i, 0]
            if x >=0 and x < res:
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

def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None, pc=True):
    if not n_avg:
        n_avg = 1000 // options.sequence_length
    activations, rate_map, g, pos = compute_ratemaps(model, trajectory_generator,
                                                     options, res=res, n_avg=n_avg, 
                                                     Ng=options.Ng, pc=pc)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    imsave(imdir + "/" + str(step) + ".png", rm_fig)


def save_autocorr(sess, model, save_name, trajectory_generator, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range=((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)
    
    res = dict()
    index_size = 100
    for _ in range(index_size):
      feed_dict = trajectory_generator.feed_dict(flags.box_width, flags.box_height)
      mb_res = sess.run({
          'pos_xy': model.target_pos,
          'bottleneck': model.g,
      }, feed_dict=feed_dict)
      res = utils.concat_dict(res, mb_res)
        
    filename = save_name + '/autocorrs_' + str(step) + '.pdf'
    imdir = flags.save_dir + '/'
    out = utils.get_scores_and_plot(
                latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                imdir, filename)

def plot_1d_performance(place_cell, generator, options, trainer):
    # check if the model generalizes well
    inputs, pc_outputs, pos = generator.get_test_batch()
    pos = pos.cpu()[:5]
    pred_pos = place_cell.get_nearest_cell_pos(trainer.predict(inputs)[0]).cpu()[:5]
    centers = place_cell.centers.cpu()
    l = options.track_length/2

    fig, axes = plt.subplots(5, 1, figsize=(13, 3), sharex=True)

    # Colormap for the time points in each trajectory
    time_cmap = plt.cm.Blues
    test_cmap = plt.cm.Reds

    # Normalize the time points for the colormap
    norm = plt.Normalize(0, pos.shape[1])

    # Plot each trajectory in its own subplot
    for i, (ax, traj) in enumerate(zip(axes, pos)):
        ax.plot(centers, torch.zeros_like(centers), 'o', color='gray', markersize=2, alpha=0.2)
        for j, point in enumerate(traj):
            ax.plot(traj[j], 0, 'o', color=time_cmap(norm(j)), markersize=5, alpha=0.6)
            ax.plot(pred_pos[i, j], 0, 'x', color=test_cmap(norm(j)), markersize=8)
        
        if i == 0:
            # add a legend
            ax.plot(-l, 0.05, 'o', color=time_cmap(norm(pos.shape[1]-1)), markersize=5, alpha=0.6)
            ax.text(-l+l/20, 0.05, 'True Position', va='center', ha='left')
            ax.plot(-l, -0.05, 'x', color=test_cmap(norm(pos.shape[1]-1)),markersize=8)
            ax.text(-l+l/20, -0.05, 'Predicted Position', va='center', ha='left')

        
        ax.set_xlim(-l-l/20, l+l/20)
        ax.set_ylim(-0.1, 0.1)
        ax.get_yaxis().set_visible(False)  # Hide the y-axis
        # ax.get_xaxis().set_visible(False)  # Hide the x-axis

    fig.text(0.5, 0.01, 'Track Position', ha='center', va='center')
    fig.suptitle('Separate Trajectories on a 1D Track')

    # add a colorbar for the time points
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    sm = plt.cm.ScalarMappable(cmap=time_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Time step')

    plt.savefig(os.path.join(options.save_dir, '1d_performance.png'))

def plot_1d_ratemaps(rate_map, options):
    n_col = 4
    fig, ax = plt.subplots(n_col, options.Ng//n_col, figsize=(2*options.Ng//n_col, n_col))
    for i, ax in enumerate(ax.flatten()):
        r = (rate_map[i] - rate_map[i].min()) / (rate_map[i].max() - rate_map[i].min())
        ax.plot(np.linspace(-options.track_length/2, options.track_length/2, 200), r, lw=3)
        # set overall x and y labels
        if i == 0:
            ax.set_ylabel('Activation')
            ax.set_title(f'Grid Cell {i+1}')
        else:
            ax.get_yaxis().set_visible(False)
            
        if i == options.Ng // n_col * (n_col - 1):
            ax.set_xlabel('Position')
        else:
            ax.get_xaxis().set_visible(False)

        ax.set_xticks([-options.track_length/2, 0, options.track_length/2])
        ax.set_xticklabels([-options.track_length/2, 0, options.track_length/2])
    plt.savefig(os.path.join(options.save_dir, '1d_ratemaps.png'))