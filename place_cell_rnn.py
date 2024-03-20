import numpy as np
import torch
import os
import time
from matplotlib import pyplot as plt

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
from visualize import *

# Training hyperparameters to fully reproduce Sorscher et al. 2023
class Options:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    oned = False
    Np = 100
    Ng = 20
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
    n_epochs = 10
    n_steps = 10
    learning_rate = 1e-4    
    weight_decay = 1e-4    
    decay_step_size = 50
    decay_rate = 0.9
    activation = 'relu'   
    restore = None # if restore, this should be the timestamp of the saved model
    preloaded_data = False
    save = False 
    save_every = 100
options = Options()

# save directory
now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
if options.restore is not None:
    now = options.restore
options.save_dir = os.path.join('./results/rnn', now)

if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)
print('Saving to:', options.save_dir)

# define place cells, trajectory generator, model, and trainer
place_cell = PlaceCells(options)
generator = TrajectoryGenerator(options, place_cell)
model = RNN(options, place_cell).to(options.device)
trainer = Trainer(options, model, generator, place_cell, restore=options.restore)

trainer.train(preloaded_data=options.preloaded_data, save=options.save)
plot_2d_performance(place_cell, generator, options, trainer)
rate_map = compute_ratemaps(model, trainer, generator, options, res=20, n_avg=200, Ng=options.Ng)
plot_2d_ratemaps(rate_map, options, n_col=4)
plot_loss_err(trainer, options)