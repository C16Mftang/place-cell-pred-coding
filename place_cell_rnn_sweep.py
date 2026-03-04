import numpy as np
import torch
import os
import time
import argparse
import json
import yaml
import wandb

from src.data.place_cells import PlaceCells
from src.data.trajectory_generator import TrajectoryGenerator
from src.model import RNN
from src.trainer import Trainer
from src.visualize import *
import src.utils as utils
from src.constants import *

# Training hyperparameters to fully reproduce Sorscher et al. 2023
parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

# Training hyperparameters to fully reproduce Sorscher et al. 2023
parser.add_argument("--Np", type=int, default=512, help="Number of place cells")
parser.add_argument("--Ng", type=int, default=1024, help="Number of grid cells")
parser.add_argument("--Nv", type=int, default=2, help="Number of velocity inputs")
parser.add_argument(
    "--DoG",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Whether to use Difference of Gaussians for place cell RFs",
)
parser.add_argument(
    "--box_width", type=float, default=1.6, help="Width of the environment box"
)
parser.add_argument(
    "--box_height", type=float, default=1.6, help="Height of the environment box"
)
parser.add_argument(
    "--sequence_length", type=int, default=10, help="Length of the trajectory sequence"
)
parser.add_argument("--dt", type=float, default=0.02, help="Time step size")
parser.add_argument(
    "--batch_size", type=int, default=500, help="Batch size for training"
)
parser.add_argument(
    "--n_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument(
    "--n_steps", type=int, default=100, help="Number of steps per epoch"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
)
parser.add_argument(
    "--rec_activation",
    type=str,
    default="relu",
    help="Recurrent activation function for the RNN",
)
parser.add_argument(
    "--out_activation",
    type=str,
    default="softmax",
    help="Output activation function for the RNN",
)
parser.add_argument(
    "--restore", type=str, default=None, help="Timestamp of the saved model to restore"
)
parser.add_argument(
    "--preloaded_data",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Whether to use preloaded data",
)
parser.add_argument(
    "--save",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Whether to save the model",
)
parser.add_argument("--loss", type=str, default="CE", help="Loss function for training")
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
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    help="Mode for running the model; input run folder name for model inspection",
)
parser.add_argument(
    "--normalize_pc",
    type=str,
    default="softmax",
    help="Transformation applied to place cells in generation",
)
parser.add_argument(
    "--truncating", type=int, default=0, help="Truncating steps for BPTT"
)
parser.add_argument(
    "--env_shape",
    type=str,
    default='rectangle',
    help="Shape of the simulated environment."
)
parser.add_argument("--place_cell_rf", type=float, default=0.12, help='diameter of place fields')
parser.add_argument("--save_every", type=int, default=50, help="Save model interval")
parser.add_argument(
    "--rf_std", type=float, default=0, help="Standard deviation of place cell RFs"
)
parser.add_argument(
    "--place_cell_rf_prob", 
    type=float, 
    default=None,
    nargs='+', 
    help="Probability of discrete place cell rfs"
)
parser.add_argument(
    "--place_cell_center_seed",
    type=int,
    default=0,
    help="Random seed for place cell center generation",
)
parser.add_argument("--decay_step_size", type=int, default=10)
parser.add_argument("--decay_rate", type=float, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument(
    "--weight_init",
    type=str,
    default="default",
    help="Weight initialization: default|kaiming_uniform|kaiming_normal",
)
parser.add_argument(
    "--init_gain",
    type=float,
    default=1.0,
    help="Gain/scale parameter used by selected weight initialization",
)

options = parser.parse_args()
options.periodic = PERIODIC
options.device = DEVICE
options.oned = ONED
# options.weight_decay = WEIGHT_DECAY
# options.decay_step_size = DECAY_STEP_SIZE
# options.decay_rate = DECAY_RATE
options.surround_scale = SURROUND_SCALE

with open("./config_rnn.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)

options.learning_rate = wandb.config.learning_rate
options.truncating = wandb.config.truncating
options.weight_decay = wandb.config.weight_decay
options.decay_rate = wandb.config.decay_rate

# define place cells, trajectory generator, model, and trainer
place_cell = PlaceCells(options)
generator = TrajectoryGenerator(options, place_cell, environment=options.env_shape)
model = RNN(options, place_cell).to(options.device)
trainer = Trainer(options, model, generator, place_cell, restore=options.restore)

trainer.train(preloaded_data=options.preloaded_data, save=options.save)
print(options)


