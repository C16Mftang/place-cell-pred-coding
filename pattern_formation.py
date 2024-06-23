# Code adapted from https://github.com/ganguli-lab/grid-pattern-formation

import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

from visualize import plot_ratemaps
from visualize import compute_grid_scores
# from utils import generate_run_ID, load_trained_weights
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
from tqdm import tqdm
import os

# Training options and hyperparameters
class Options:
    pass
options = Options()

options.save_dir = './results/pattern_formation/'
options.n_epochs = 5          # number of training epochs
options.n_steps = 1000        # batches per epoch
options.batch_size = 200      # number of trajectories per batch
options.sequence_length = 20  # number of steps in trajectory
options.learning_rate = 1e-4  # gradient descent learning rate
options.Np = 512              # number of place cells
options.Ng = 4096             # number of grid cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.RNN_type = 'RNN'      # RNN or LSTM
options.activation = 'relu'   # recurrent nonlinearity
options.weight_decay = 1e-4   # strength of weight decay on recurrent weights
options.DoG = True            # use difference of gaussians tuning curves
options.periodic = False      # trajectories with periodic boundary conditions
options.box_width = 1.8      # width of training environment
options.box_height = 1.8      # height of training environment
options.device = 'cuda'
options.normalize_pc = 'softmax'
options.dt = 0.02
options.type = 'multiple'
# options.run_ID = generate_run_ID(options)

save_dir = os.path.join(options.save_dir, options.type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

place_cells = PlaceCells(options)
res = 55
C = place_cells.compute_covariance(res=res)

print(C.shape)

# Fourier transform
Ctilde = np.fft.fft2(C)
Ctilde[0,0] = 0

def convolve_with_C(g, Ctilde):
    '''
    Convolves the input g with the kernel C
    '''
    gtilde = np.fft.fft2(g, [res, res])
    gconv = np.real(np.fft.ifft2(gtilde*Ctilde))
    gconv = np.roll(np.roll(gconv, res//2+1, axis=1), res//2+1, axis=2)
    
    return gconv

# Symmetry-breaking nonlinearity (relu)
def phi(x,r):
    return np.maximum(r*x,0)

def plot_grid_cells(res, Ng, G, options):
    idx, scores = compute_grid_scores(res, G, options)
    sorted_gcs = G[idx]
    np.savez(
        os.path.join(save_dir, 'sorted_gcs.npz'), 
        gcs=sorted_gcs, 
        scores=scores,
    )

    n = int(np.sqrt(Ng))
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for i, ax in enumerate(axes.flatten()):
        gc = (sorted_gcs[i] - np.min(sorted_gcs[i])) / (np.max(sorted_gcs[i]) - np.min(sorted_gcs[i]))
        ax.imshow(gc, cmap='jet')
        ax.set_title(f'{scores[i]:.2f}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_cells.png'))

Ng = 256
r = 30
T = 2000
lr = 1e-3
G = np.random.randn(Ng,res,res) * 1e-8

if options.type == 'single':
    print('Training single grid cell...')
    for i in tqdm(range(T)):
        G +=lr*(-G + convolve_with_C(G, Ctilde))
        G = phi(G, 1)
        
    plot_grid_cells(res, Ng, G, options)

else:
    print('Training multiple grid cells...')
    for i in tqdm(range(T)):
        H = convolve_with_C(G, Ctilde)
        Hr = H.reshape([Ng, -1])
        Gr = G.reshape([Ng, -1])
        oja = Gr.T.dot(np.tril(Gr.dot(Hr.T))).T.reshape([Ng,res,res])

        G += lr * (H - oja + phi(G,r))
        
    plot_grid_cells(res, Ng, G, options)

