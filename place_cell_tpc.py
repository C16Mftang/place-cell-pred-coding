import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import HierarchicalPCN, TemporalPCN
from trainer import PCTrainer
from visualize import *

class Options:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    oned = False
    Np = 500
    Ng = 50
    Nv = 2
    DoG = False
    box_width = 1.6         
    box_height = 1.6 
    place_cell_rf = 0.12
    surround_scale = 2
    periodic = False
    sequence_length = 20
    dt = 0.02
    batch_size = 500
    n_epochs = 100
    n_steps = 100
    learning_rate = 2e-3
    weight_decay = 1e-3
    decay_step_size = 50
    decay_rate = 0.9 
    lambda_z = 0. 
    lambda_z_init = 0. 
    inf_iters = 20 
    test_inf_iters = 300 
    inf_lr = 1e-2
    out_activation = utils.Softmax()
    rec_activation = utils.Tanh()
    restore = None # if restore, this should be the timestamp of the saved model
    preloaded_data = False
    save = False
    save_every = 100
options = Options()

# save directory
now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
if options.restore is not None:
    now = options.restore
options.save_dir = os.path.join('./results/tpc', now)

if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)
print('Saving to:', options.save_dir)

# define place cells, trajectory generator, model, and trainer
place_cell = PlaceCells(options)
generator = TrajectoryGenerator(options, place_cell)
model = TemporalPCN(options).to(options.device)
init_model = HierarchicalPCN(options).to(options.device)
trainer = PCTrainer(options, model, init_model, generator, place_cell, restore=options.restore)

trainer.train(preloaded_data=options.preloaded_data, save=options.save)
plot_place_cells(place_cell, options, res=30)
plot_2d_performance(place_cell, generator, options, trainer)
rate_map = compute_ratemaps(model, trainer, generator, options, res=20, n_avg=200, Ng=options.Ng)
plot_2d_ratemaps(rate_map, options, n_col=4)
plot_loss_err(trainer, options)
utils.save_options_to_json(options, os.path.join(options.save_dir, 'configs.json'))