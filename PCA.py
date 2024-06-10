import numpy as np
import torch
import os
from matplotlib import pyplot as plt

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import *
from visualize import compute_grid_scores
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA, NMF, SparsePCA

# Training options and hyperparameters
class Options:
    pass
options = Options()

options.batch_size = 100      # number of trajectories per batch
options.Np = 512              # number of place cells
options.Ng = 64           # number of grid cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.DoG = True  # use difference of gaussians tuning curves
options.periodic = False  # trajectories with periodic boundary conditions
options.box_width = 1.4    # width of training environment
options.box_height = 1.4      # height of training environment
options.device = 'cuda'     # specify devices
options.normalize_pc = 'softmax'
options.res = 30

save_dir = './results/pca/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

pos = np.array(
    np.meshgrid(
        np.linspace(-options.box_width/2, options.box_width/2, options.res),
        np.linspace(-options.box_height/2, options.box_height/2, options.res)
    )
).T
pos = torch.tensor(pos).to(options.device)

# get place cell activations
pc_outputs = place_cells.get_activation(pos).detach().cpu()
pc_outputs = pc_outputs.reshape(options.res ** 2, options.Np)
pc_outputs_np = pc_outputs.cpu().numpy() # Nx x Np

def sanger_update(W, x, eta):
    y = np.dot(W, x)  # Calculate the output
    W_delta = eta * (np.outer(y, x) - np.tril(np.outer(y, y)) @ W)
    W += W_delta
    # Normalize the weight vectors
    W = W * (W > 0)  # ReLU
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W

P = pc_outputs_np.copy()

# Normalize P to have zero mean across samples
P -= np.mean(P, axis=0)

# Parameters
W = np.random.randn(options.Ng, options.Np) # Weight matrix W of size k x p
W /= np.linalg.norm(W, axis=1, keepdims=True)  # Normalize the weight vectors
W_old = W.copy()
eta = 1  # Learning rate
epochs = 500  # Number of epochs

# Training
diffs = []
for epoch in tqdm(range(epochs)):
    if (epoch + 1) % 10 == 0:
        eta *= 1
    for x in P:
        W = sanger_update(W, x, eta)
    
    diff = np.linalg.norm(W - W_old)
    if diff < 1e-6:
        break
    W_old = W.copy()
    diffs.append(diff)

# Plot the convergence]
plt.figure()
plt.plot(diffs[5:])
plt.savefig(save_dir + 'convergence.png')

y = np.dot(W, P.T)
gcs = y.reshape((-1, options.res, options.res))
idx, scores = compute_grid_scores(options.res, gcs, options)
sorted_gcs = gcs[idx]
np.savez(save_dir + 'gc_and_scores.npz', gcs=sorted_gcs, scores=scores)

n = int(np.sqrt(options.Ng))
fig, axes = plt.subplots(n, n, figsize=(n, n))
for i, ax in enumerate(axes.flatten()):
    gc = (sorted_gcs[i] - np.min(sorted_gcs[i])) / (np.max(sorted_gcs[i]) - np.min(sorted_gcs[i]))
    ax.imshow(gc, cmap='jet')
    ax.set_title(f'{scores[i]:.2f}')
    ax.axis('off')
plt.tight_layout()
plt.savefig(save_dir + 'sorted_gcs.png')