import numpy as np
import torch
import os

from tqdm import tqdm
from matplotlib import pyplot as plt

# from utils import generate_run_ID, load_trained_weights
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from get_data import get_traj_loader, generate_traj_data
from model import RNN
from trainer import Trainer
from utils import generate_run_ID

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

# Training hyperparameters to fully reproduce Sorscher et al. 2023
class Options:
    pass
options = Options()

options.full_size = 50000
options.n_epochs = 200          # number of training epochs
options.n_steps = 1000          # number of batches in one epoch
options.batch_size = 200        # number of trajectories per batch
options.sequence_length = 20    # number of steps in trajectory
options.learning_rate = 1e-4    # gradient descent learning rate
options.Np = 512                # number of place cells
options.Ng = 4096               # number of grid cells
options.place_cell_rf = 0.12    # width of place cell center tuning curve (m)
options.surround_scale = 2      # if DoG, ratio of sigma2^2 to sigma1^2
options.RNN_type = 'RNN'        # RNN or LSTM
options.activation = 'relu'     # recurrent nonlinearity
options.weight_decay = 1e-4     # strength of weight decay on recurrent weights
options.DoG = True              # use difference of gaussians tuning curves
options.periodic = False        # trajectories with periodic boundary conditions
options.box_width = 2.2         # width of training environment
options.box_height = 2.2        # height of training environment
options.device = device
options.save_dir = 'models/'
options.data_source = 'online'
options.run_ID = generate_run_ID(options)
options.decay_step_size = 5
options.decay_rate = 1

# define place cells, trajectory generator, model, and trainer
place_cells = PlaceCells(options)
model = RNN(options, place_cells).to(device)
trajectory_generator = TrajectoryGenerator(options, place_cells)
trainer = Trainer(options, model, trajectory_generator)

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
plt.savefig(os.path.join(options.save_dir, options.run_ID)+'/loss')