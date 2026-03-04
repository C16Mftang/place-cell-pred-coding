import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
import argparse
import json
import yaml
import wandb

from src.data.place_cells import PlaceCells
from src.data.trajectory_generator import TrajectoryGenerator
from src.model import HierarchicalPCN, TemporalPCN
from src.trainer import PCTrainer
from src.visualize import *
import src.utils as utils
from src.constants import *

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--Np", type=int, default=512, help="Number of place cells")
parser.add_argument("--Ng", type=int, default=1024, help="Number of grid cells")
parser.add_argument("--Nv", type=int, default=2, help="Number of velocity inputs")
parser.add_argument(
    "--DoG",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Use Difference of Gaussians for place cell receptive fields",
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
parser.add_argument("--dt", type=float, default=0.02, nargs='+', help="Time step size")
parser.add_argument(
    "--batch_size", type=int, default=500, help="Batch size for training"
)
parser.add_argument(
    "--n_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument(
    "--n_steps", type=int, default=100, help="Number of batches"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for optimization"
)
parser.add_argument(
    "--inf_iters", type=int, default=10, help="Number of inference iterations"
)
parser.add_argument(
    "--test_inf_iters",
    type=int,
    default=20,
    help="Number of inference iterations for testing",
)
parser.add_argument(
    "--inf_lr", type=float, default=1e-2, help="Learning rate for inference"
)
parser.add_argument(
    "--out_activation",
    type=str,
    default="softmax",
    help="Activation function for the output layer",
)
parser.add_argument(
    "--rec_activation",
    type=str,
    default="relu",
    help="Activation function for the recurrent layer",
)
parser.add_argument(
    "--restore", type=str, default=None, help="Timestamp of the saved model to restore"
)
parser.add_argument(
    "--preloaded_data",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Use preloaded data for training",
)
parser.add_argument(
    "--save",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Save the trained model",
)
parser.add_argument("--loss", type=str, default="CE", help="Loss function for training")
parser.add_argument(
    "--normalize_pc",
    type=str,
    default="softmax",
    help="transformation applied to place cells in generation",
)
parser.add_argument(
    "--is_wandb",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Use wandb for logging",
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
    "--no_velocity",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Without velocity input",
)
parser.add_argument(
    "--env_shape",
    type=str,
    default='rectangle',
    help="Shape of the simulated environment."
)
parser.add_argument("--save_every", type=int, default=50, help="Save model interval")
parser.add_argument("--place_cell_rf", type=float, default=0.12, help='diameter of place fields')
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

options = parser.parse_args()
options.periodic = PERIODIC
options.device = DEVICE
options.oned = ONED
options.weight_decay = WEIGHT_DECAY
options.decay_step_size = DECAY_STEP_SIZE
options.decay_rate = DECAY_RATE
options.lambda_z = LAMBDA_Z
options.lambda_z_init = LAMBDA_Z_INIT
options.surround_scale = SURROUND_SCALE

with open("./config_tpc.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)

options.learning_rate = wandb.config.learning_rate
options.inf_lr = wandb.config.inf_lr

place_cell = PlaceCells(options)
generator = TrajectoryGenerator(options, place_cell)
model = TemporalPCN(options).to(options.device)
init_model = HierarchicalPCN(options).to(options.device)
trainer = PCTrainer(options, model, init_model, generator, place_cell, restore=options.restore)

trainer.train(preloaded_data=options.preloaded_data, save=options.save)
print(options)


