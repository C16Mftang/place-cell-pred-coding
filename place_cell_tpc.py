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
    Np = 512
    Ng = 4096
    Nv = 2
    DoG = True
    box_width = 1.6         
    box_height = 1.6 
    place_cell_rf = 0.12
    surround_scale = 2
    periodic = True
    sequence_length = 20
    dt = 0.02
    batch_size = 500
    n_epochs = 400
    n_steps = 100
    learning_rate = 1e-4
    weight_decay = 0
    decay_step_size = 50
    decay_rate = 0.9 
    lambda_z = 0.
    lambda_z_init = 0.1
    inf_iters = 20 
    test_inf_iters = 20
    inf_lr = 1e-2
    out_activation = utils.Tanh()
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

utils.save_options_to_json(options, os.path.join(options.save_dir, 'configs.json'))

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