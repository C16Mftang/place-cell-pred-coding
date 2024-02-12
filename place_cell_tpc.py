import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from tqdm import tqdm
from matplotlib import pyplot as plt

import predictive_coding as pc
# from utils import generate_run_ID, load_trained_weights
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from get_data import get_traj_loader, generate_traj_data
from model import RNN
from trainer import Trainer
from utils import generate_run_ID

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


class tPCLayer(nn.Module):
    def __init__(self, size_in, size_hidden):
        super(tPCLayer, self).__init__()
        # Initialize the Win and Wr linear layers according to the specified shapes
        if size_in > 0:
            self.Win = nn.Linear(size_in, size_hidden, bias=False)
        self.Wr = nn.Linear(size_hidden, size_hidden, bias=False)
        # determine whether there is an external input
        self.is_in = True if size_in > 0 else False

    def forward(self, inputs):
        # input: a tuple of two tensors (input, hidden (from the previous time step))
        # Compute Win(input) + Wr(hidden) and return the result
        return self.Win(inputs[1]) + self.Wr(inputs[0]) if self.is_in else self.Wr(inputs[0])

class InitEncoder(nn.Module):
    def __init__(self, size_v, size_p, size_hidden):
        super(InitEncoder, self).__init__()
        # Initialize the Win and Wr linear layers according to the specified shapes
        self.Wv = nn.Linear(size_v, size_hidden, bias=False)
        self.Wp = nn.Linear(size_p, size_hidden, bias=False)

    def forward(self, inputs):
        # input: a tuple of two tensors (velocity, init_pos)
        # Compute Win(input) + Wr(hidden) and return the result
        return self.Wv(inputs[0]) + self.Wp(inputs[1])

def mse_loss(output, _target):
    return (output - _target).pow(2).sum() * 0.5

def ce_loss(output, _target):
    pred = F.softmax(output, dim=-1)
    return -(_target * torch.log(pred)).sum(-1).mean()


# Training hyperparameters to fully reproduce Sorscher et al. 2023
class Options:
    pass
options = Options()

options.n_epochs = 500          # number of training epochs
options.n_steps = 20          # number of batches in one epoch
options.batch_size = 100        # number of trajectories per batch
options.sequence_length = 20    # number of steps in trajectory
options.learning_rate = 1e-4    # gradient descent learning rate
options.size_in = 2             # dimension of velocity input
options.Np = 512                # number of place cells
options.Ng = 2048               # number of grid cells
options.place_cell_rf = 0.12    # width of place cell center tuning curve (m)
options.surround_scale = 2      # if DoG, ratio of sigma2^2 to sigma1^2
options.RNN_type = 'tPC'        # RNN or LSTM
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
activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
}

# define place cells, trajectory generator
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

init_layer = nn.Sequential(
    InitEncoder(options.size_in, options.Np, options.Ng),
    pc.PCLayer(),
    activations[options.activation],
)

rec_layer = nn.Sequential(
    tPCLayer(options.size_in, options.Ng),
    pc.PCLayer(),
    activations[options.activation],
)

forward_layer = nn.Sequential(
    nn.Linear(options.Ng, options.Np),
    pc.PCLayer(),
    # activations[options.activation],
)

# to get the initial grid cell
encoder = nn.Sequential(
    init_layer,
    forward_layer,
).to(device)

model = nn.Sequential(
    rec_layer,
    forward_layer,
).to(device)

model.train()
encoder.train()

enc_trainer = pc.PCTrainer(
    encoder, 
    T=200,
    update_x_at='all',
    optimizer_x_fn=optim.SGD,
    optimizer_x_kwargs={"lr": 1e-2},
    update_p_at='last',
    optimizer_p_fn=optim.Adam,
    optimizer_p_kwargs={"lr": options.learning_rate},
    plot_progress_at=[],
)

tpc_trainer = pc.PCTrainer(
    model, 
    T=20,
    update_x_at='all',
    optimizer_x_fn=optim.SGD,
    optimizer_x_kwargs={"lr": 1e-2},
    update_p_at='last',
    optimizer_p_fn=optim.Adam,
    optimizer_p_kwargs={"lr": options.learning_rate},
    plot_progress_at=[],
)

gen = trajectory_generator.get_generator()

losses = []
for i in range(options.n_epochs):
    epoch_loss = 0
    for step_idx in range(options.n_steps):
        inputs, pc_outputs, pos = next(gen)
        v, p0 = inputs
        assert options.sequence_length == v.shape[0], "sequence length not match"
        for k in range(options.sequence_length):
            if k == 0:
                results = enc_trainer.train_on_batch(
                    inputs=(v[k], p0),
                    loss_fn=ce_loss,
                    loss_fn_kwargs={"_target": pc_outputs[k]},
                    is_log_progress=False,
                    is_return_results_every_t=False,
                )
                hidden = encoder[0][1].get_x().detach().clone()
            else:
                result = tpc_trainer.train_on_batch(
                    inputs=(hidden, v[k]),
                    loss_fn=ce_loss,
                    loss_fn_kwargs={"_target": pc_outputs[k]}, # can add weight decay here
                    is_log_progress=False,
                    is_return_results_every_t=False,
                )
                hidden = model[0][1].get_x().detach().clone()
            # log the output loss
            epoch_loss += results["loss"][0]
    epoch_loss /= (options.sequence_length * options.n_steps)
    losses.append(epoch_loss)
    print(f'epoch{i}, loss: {epoch_loss}')