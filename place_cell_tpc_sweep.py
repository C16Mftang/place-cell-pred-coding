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

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import HierarchicalPCN, TemporalPCN
from trainer import PCTrainer
from visualize import *
import utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--oned', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use one-dimensional place cells')
parser.add_argument('--Np', type=int, default=512, help='Number of place cells')
parser.add_argument('--Ng', type=int, default=2048, help='Number of grid cells')
parser.add_argument('--Nv', type=int, default=2, help='Number of velocity inputs')
parser.add_argument('--DoG', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use Difference of Gaussians for place cell receptive fields')
parser.add_argument('--box_width', type=float, default=1.6, help='Width of the environment box')
parser.add_argument('--box_height', type=float, default=1.6, help='Height of the environment box')
parser.add_argument('--place_cell_rf', type=float, default=0.12, help='Place cell receptive field size')
parser.add_argument('--surround_scale', type=int, default=2, help='Scale factor for the surround inhibition')
parser.add_argument('--periodic', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use periodic boundary conditions')
parser.add_argument('--sequence_length', type=int, default=20, help='Length of the trajectory sequence')
parser.add_argument('--dt', type=float, default=0.02, help='Time step size')
parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--n_steps', type=int, default=100, help='Number of steps in each trajectory')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimization')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimization')
parser.add_argument('--decay_step_size', type=int, default=50, help='Step size for learning rate decay')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for learning rate decay')
parser.add_argument('--lambda_z', type=float, default=0, help='Weight for the regularization term')
parser.add_argument('--lambda_z_init', type=float, default=0, help='Initial weight for the regularization term')
parser.add_argument('--inf_iters', type=int, default=20, help='Number of inference iterations')
parser.add_argument('--test_inf_iters', type=int, default=20, help='Number of inference iterations for testing')
parser.add_argument('--inf_lr', type=float, default=2e-2, help='Learning rate for inference')
parser.add_argument('--out_activation', type=str, default='tanh', help='Activation function for the output layer')
parser.add_argument('--rec_activation', type=str, default='tanh', help='Activation function for the recurrent layer')
parser.add_argument('--restore', type=str, default=None, help='Timestamp of the saved model to restore')
parser.add_argument('--preloaded_data', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use preloaded data for training')
parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default=False, help='Save the trained model')
parser.add_argument('--save_every', type=int, default=100, help='Save the model every n epochs')
parser.add_argument('--loss', type=str, default='MSE', help='Loss function for training')
parser.add_argument('--normalize_pc', type=str, default='softmax', help='transformation applied to place cells in generation')
parser.add_argument('--is_wandb', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use wandb for logging')
parser.add_argument('--sweep', type=lambda x: (str(x).lower() == 'true'), default=True, help='Hyperparameter tune')
options = parser.parse_args()

# save directory
# now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
# if options.restore is not None:
#     now = options.restore
# options.save_dir = os.path.join('./results/tpc', now)

# if not os.path.exists(options.save_dir):
#     os.makedirs(options.save_dir)
# print('Saving to:', options.save_dir)

# utils.save_options_to_json(options, os.path.join(options.save_dir, 'configs.json'))

# define place cells, trajectory generator, model, and trainer
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


