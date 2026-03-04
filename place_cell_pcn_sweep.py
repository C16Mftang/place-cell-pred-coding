import numpy as np
import torch
import os
from matplotlib import pyplot as plt
import argparse
import time
import json
import wandb
import yaml
from tqdm import tqdm

from src.data.place_cells import PlaceCells
from src.data.trajectory_generator import TrajectoryGenerator

from src.model import *
from src.visualize import compute_grid_scores, plot_all_ratemaps

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training options and hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--save_dir",
    type=str,
    default="./results/pcn/",
    help="Directory to save the models",
)
parser.add_argument("--Np", type=int, default=512, help="Number of place cells")
parser.add_argument("--Ng", type=int, default=1024, help="Number of grid cells")
parser.add_argument(
    "--place_cell_rf",
    type=float,
    default=0.12,
    help="Width of place cell center tuning curve (m)",
)
parser.add_argument(
    "--surround_scale",
    type=int,
    default=2,
    help="If DoG, ratio of sigma2^2 to sigma1^2",
)
parser.add_argument(
    "--DoG",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Use difference of gaussians tuning curves",
)
parser.add_argument(
    "--periodic",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Trajectories with periodic boundary conditions",
)
parser.add_argument(
    "--box_width", type=float, default=1.6, help="Width of training environment"
)
parser.add_argument(
    "--box_height", type=float, default=1.6, help="Height of training environment"
)
parser.add_argument(
    "--env_shape",
    type=str,
    default='rectangle',
    help="Shape of the simulated environment."
)
parser.add_argument(
    "--out_activation", type=str, default="linear", help="Nonlinearity function"
)
parser.add_argument(
    "--inf_lr", type=float, default=1e-2, help="Learning rate for inference"
)
parser.add_argument(
    "--inf_lr_test",
    type=float,
    default=1e-3,
    help="Learning rate for inference during testing",
)
parser.add_argument(
    "--inf_iters", type=int, default=10, help="Number of inference iterations"
)
parser.add_argument(
    "--inf_iters_test",
    type=int,
    default=10,
    help="Number of inference iterations during test",
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
)
parser.add_argument(
    "--n_epochs", type=int, default=100, help="Number of training iterations"
)
parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
parser.add_argument("--res", type=int, default=30, help="Sample resolution")
parser.add_argument(
    "--decay_step_size", type=int, default=10, help="Step size for learning rate decay"
)
parser.add_argument(
    "--decay_rate", type=float, default=0.9, help="Decay rate for learning rate"
)
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--loss", type=str, default="MSE", help="Loss function")
parser.add_argument(
    "--lambda_z_init", type=float, default=0.1, help="Lambda for hidden layer"
)
parser.add_argument(
    "--normalize_pc",
    type=str,
    default="softmax",
    help="transformation applied to place cells in generation",
)
parser.add_argument(
    "--relu_inf",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Use ReLU for inference",
)
parser.add_argument(
    "--rf_std", type=float, default=0, help="Standard deviation of place cell RFs"
)
parser.add_argument("--save_every", type=int, default=2, help="Save model interval")
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument(
    "--place_cell_rf_prob", 
    type=float, 
    default=None,
    nargs='+', 
    help="Probability of discrete place cell rfs"
)
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    help="Mode for running the model; input run folder name for model inspection",
)
parser.add_argument(
    "--is_wandb",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Whether to use wandb for logging",
)
parser.add_argument(
    "--sweep",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Hyperparameter tune",
)
options = parser.parse_args()

with open("./config_pcn.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)
options.learning_rate = wandb.config.learning_rate
options.inf_lr = wandb.config.inf_lr

place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

usx = np.random.uniform(
    -options.box_width / 2, options.box_width / 2, (options.sample_size,)
)
usy = np.random.uniform(
    -options.box_width / 2, options.box_width / 2, (options.sample_size,)
)
pos = np.vstack([usx, usy]).T
pos = pos[:, None, ...]
pos = torch.tensor(pos).to(device)

# get place cell activations
pc_outputs = place_cells.get_activation(pos).detach().cpu()
pc_outputs = pc_outputs.reshape(-1, options.Np)
pc_outputs = pc_outputs - pc_outputs.mean(dim=0, keepdim=True)

# Train the PCN
nodes = [options.Ng, options.Np]
pcn = MultilayerPCN(
    nodes,
    options.out_activation,
    lamb=options.lambda_z_init,
    use_bias=False,
    relu_inf=options.relu_inf,
).to(device)
X = pc_outputs.to(device)
optimizer = torch.optim.Adam(
    pcn.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=options.decay_step_size, gamma=options.decay_rate
)

train_mses = []
norms = []
if options.is_wandb == True and options.sweep == False:
    wandb.init(project='place-cell-pcn', config=options)
for epoch in range(options.n_epochs):
    iter_loss = 0
    tbar = tqdm(range(0, options.sample_size, options.batch_size))
    for batch_idx in tbar:
        data = X[batch_idx : batch_idx + options.batch_size]
        optimizer.zero_grad()
        pcn.inference(data, options.inf_iters, options.inf_lr)
        loss = pcn.get_energy()
        loss.backward()
        optimizer.step()
        iter_loss += loss.item()

        tbar.set_description(
            "Epoch: {}/{}. Loss: {}".format(
                epoch + 1, options.n_epochs, np.round(loss.item(), 8),
            )
        )

    train_mse = iter_loss / (options.sample_size / options.batch_size)
    if options.is_wandb:
        wandb.log({
            'energy': train_mse,
        })
    train_mses.append(train_mse)
    scheduler.step()

if options.is_wandb:
    wandb.finish()

