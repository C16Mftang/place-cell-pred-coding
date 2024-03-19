import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import os
import pickle
import time

from place_cells import PlaceCells1D
from trajectory_generator import TrajectoryGenerator1D

from model import *
from trainer import PCTrainer
from visualize import *
import utils

class Options():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    oned = True
    Np = 100
    Ng = 20
    Nv = 1
    track_length = 1.0
    place_cell_rf = 0.05 # None for one-hot encoding of place cells
    plce_cell_seed = 0
    periodic = False
    seq_len = 20
    dt = 0.01
    batch_size = 500
    n_epochs = 1
    n_steps = 10
    learning_rate = 2e-3
    weight_decay = 1e-3
    decay_step_size = 10
    decay_rate = 0.9 
    lambda_z = 0. # sparse penalty for the temporal model
    lambda_z_init = 0. # sparse penalty for the initial static model
    inf_iters = 20 
    test_inf_iters = 300 # inference iterations during testing, for the initial static model
    inf_lr = 1e-2
    out_activation = utils.Softmax()
    rec_activation = utils.Tanh()
options = Options()

# visualization save directory
now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
options.save_dir = os.path.join('./results/1d_tpc', now)

if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)

place_cell = PlaceCells1D(options)
generator = TrajectoryGenerator1D(options, place_cell)

model = TemporalPCN(options).to(options.device)
init_model = HierarchicalPCN(options).to(options.device)
trainer = PCTrainer(options, model, init_model, generator, place_cell, restore=False)

trainer.train(save=True, preloaded_data=True)

# visualize the performance on test set
plot_1d_performance(place_cell, generator, options, trainer)

# plot grid cells
rate_map = compute_1d_ratemaps(model, trainer, generator, options, n_avg=200, res=200, Ng=options.Ng)
plot_1d_ratemaps(rate_map, options)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(trainer.loss)
axes[0].set_title('Loss')
axes[0].set_xlabel('Training Step')
axes[0].set_ylabel('Loss')
axes[1].plot(trainer.err)
axes[1].set_title('Decoding Error')
axes[1].set_xlabel('Training Step')
axes[1].set_ylabel('Error (cm)')
plt.savefig(os.path.join(options.save_dir, 'loss.png'))

utils.save_options_to_json(options, os.path.join(options.save_dir, 'options.json'))