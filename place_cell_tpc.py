import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import os
import pickle
import time

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import *
from trainer import PCTrainer
from visualize import *
import utils

class Options():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    oned = False
    Np = 100
    Ng = 20
    Nv = 2
    DoG = False
    box_width = 1.6         # width of training environment
    box_height = 1.6 
    place_cell_rf = 0.12 # None for one-hot encoding of place cells
    surround_scale = 2
    plce_cell_seed = 0
    periodic = False
    sequence_length = 20
    dt = 0.01
    batch_size = 500
    n_epochs = 10
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

# save directory
now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
options.save_dir = os.path.join('./results/tpc', now)

if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)
print('Saving to:', options.save_dir)

# define place cells, trajectory generator
place_cell = PlaceCells(options)
generator = TrajectoryGenerator(options, place_cell)
model = TemporalPCN(options).to(options.device)
init_model = HierarchicalPCN(options).to(options.device)
trainer = PCTrainer(options, model, init_model, generator, place_cell, restore=False)

trainer.train(save=False, preloaded_data=False)
plot_2d_performance(place_cell, generator, options, trainer)
rate_map = compute_ratemaps(model, trainer, generator, options, res=20, n_avg=200, Ng=options.Ng)
plot_2d_ratemaps(rate_map, options, n_col=4)