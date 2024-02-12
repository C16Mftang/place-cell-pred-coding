import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time

from tqdm import tqdm
from matplotlib import pyplot as plt

import predictive_coding as pc
# from utils import generate_run_ID, load_trained_weights
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from get_data import get_traj_loader, generate_traj_data
from model import RNN, tPC, Bias
from trainer import Trainer
from utils import generate_run_ID
from visualize import save_ratemaps

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

def mse_loss(output, _target):
    return (output - _target).pow(2).sum() * 0.5

def ce_loss(output, _target):
    pred = F.softmax(output, dim=-1)
    return -(_target * torch.log(pred)).sum(-1).mean()


# Training hyperparameters to fully reproduce Sorscher et al. 2023
class Options:
    pass
options = Options()

options.n_epochs = 10          # number of training epochs
options.n_steps = 2          # number of batches in one epoch
options.batch_size = 10        # number of trajectories per batch
options.sequence_length = 20    # number of steps in trajectory
options.learning_rate = 1e-3    # gradient descent learning rate
options.size_in = 2             # dimension of velocity input
options.Np = 512                # number of place cells
options.Ng = 256               # number of grid cells
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
options.save_every = 2

ckpt_dir = os.path.join(options.save_dir, options.run_ID)
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
print("Saving to: {}".format(ckpt_dir))

# define place cells, trajectory generator
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)
model = tPC(options).to(device)
model.train()

# init_trainer = pc.PCTrainer(
#     model.init_model,
#     T=20,
#     update_x_at='all',
#     optimizer_x_fn=optim.SGD,
#     optimizer_x_kwargs={"lr": 1e-2},
#     update_p_at='last',
#     optimizer_p_fn=optim.Adam,
#     optimizer_p_kwargs={"lr": options.learning_rate},
#     plot_progress_at=[],
# )

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
    print(f'Epoch {i+1}/{options.n_epochs}')
    start = time.time()
    epoch_loss = 0
    epoch_err = 0
    for step_idx in range(options.n_steps):
        inputs, pc_outputs, pos = next(gen)
        v, p0 = inputs

        assert options.sequence_length == v.shape[0], "sequence length not match"

        hidden = p0 @ trajectory_generator.get_hidden_projector()
        init_state = hidden.clone()
        for k in range(options.sequence_length):
            results = tpc_trainer.train_on_batch(
                inputs=(hidden, v[k]),
                loss_fn=ce_loss,
                loss_fn_kwargs={"_target": pc_outputs[k]}, # can add weight decay here
                is_log_progress=False,
                is_return_results_every_t=False,
            )
            hidden = model.tpc[0][1].get_x().detach()
            # log the output loss
            epoch_loss += results["loss"][0]

        # log the decoded position error
        pred = model.predict(v, init_state)
        pred_pos = place_cells.get_nearest_cell_pos(pred)
        epoch_err += torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()
    
    epoch_loss /= (options.sequence_length * options.n_steps)
    epoch_err /= (options.n_steps)
    losses.append(epoch_loss)
    print(f'loss: {epoch_loss}, decoded position error: {epoch_err} m, time: {time.time() - start}')

    if (i + 1) % options.save_every == 0:
        torch.save(
            model.state_dict(), 
            os.path.join(ckpt_dir, f'epoch_{i+1}.pth')
        )
        save_ratemaps(model, trajectory_generator, options, i+1, res=20, n_avg=1000, pc=True)
# predict after training
