import numpy as np
import torch
import os
import time

from tqdm import tqdm
from matplotlib import pyplot as plt

# from utils import generate_run_ID, load_trained_weights
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from get_data import get_traj_loader, generate_traj_data
from model import RNN
from trainer import Trainer
from utils import generate_run_ID

# Training hyperparameters to fully reproduce Sorscher et al. 2023
class Options:
    full_size = 50000
    n_epochs = 500          # number of training epochs
    n_steps = 2000          # number of batches in one epoch
    batch_size = 100        # number of trajectories per batch
    sequence_length = 20    # number of steps in trajectory
    learning_rate = 1e-4    # gradient descent learning rate
    Np = 200                # number of place cells
    Ng = 50               # number of grid cells
    place_cell_rf = 0.12    # width of place cell center tuning curve (m)
    surround_scale = 2      # if DoG, ratio of sigma2^2 to sigma1^2
    RNN_type = 'RNN'        # RNN or LSTM
    activation = 'relu'     # recurrent nonlinearity
    weight_decay = 1e-4     # strength of weight decay on recurrent weights
    DoG = True              # use difference of gaussians tuning curves
    periodic = False        # trajectories with periodic boundary conditions
    box_width = 1.6         # width of training environment
    box_height = 1.6        # height of training environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_source = 'online'
    decay_step_size = 5
    decay_rate = 1
options = Options()

# save directory
now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
options.save_dir = os.path.join('./results/rnn', now)

if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)
print('Saving to:', options.save_dir)


# define place cells, trajectory generator, model, and trainer
place_cells = PlaceCells(options)
model = RNN(options, place_cells).to(options.device)
trajectory_generator = TrajectoryGenerator(options, place_cells)
trainer = Trainer(options, model, trajectory_generator, restore=False)

if options.data_source == 'pre':
    path = f'data/trajectory_{options.full_size}_{options.sequence_length}_{options.Np}.npz'
    # check if the file exists
    if os.path.exists(path):
        print('Loading pre-generated data')
    else:
        print('Generating new data')
        generate_traj_data(options)

    dataloader = get_traj_loader(path, options)
    trainer.train_batch(dataloader, n_epochs=options.n_epochs, save=True)
else:
    trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps, save=True)

plt.figure(figsize=(12,3))
plt.subplot(121)
plt.plot(trainer.err, c='black')

plt.title('Decoding error (m)'); plt.xlabel('train step')
plt.subplot(122)
plt.plot(trainer.loss, c='black');
plt.title('Loss'); plt.xlabel('train step');
plt.savefig(os.path.join(options.save_dir, 'loss'))