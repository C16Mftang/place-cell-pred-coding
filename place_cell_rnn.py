import numpy as np
import torch
import os
import time
import argparse
import json

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
from visualize import *
from scores import border_score

# Training hyperparameters to fully reproduce Sorscher et al. 2023
parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

# Training hyperparameters to fully reproduce Sorscher et al. 2023
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
parser.add_argument('--oned', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use one-dimensional place cells')
parser.add_argument('--Np', type=int, default=512, help='Number of place cells')
parser.add_argument('--Ng', type=int, default=2048, help='Number of grid cells')
parser.add_argument('--Nv', type=int, default=2, help='Number of velocity inputs')
parser.add_argument('--DoG', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use Difference of Gaussians for place cell RFs')
parser.add_argument('--box_width', type=float, default=1.4, help='Width of the environment box')
parser.add_argument('--box_height', type=float, default=1.4, help='Height of the environment box')
parser.add_argument('--place_cell_rf', type=float, default=0.12, help='Place cell receptive field size')
parser.add_argument('--surround_scale', type=int, default=2, help='Scale factor for the surround inhibition')
parser.add_argument('--periodic', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether the environment is periodic')
parser.add_argument('--sequence_length', type=int, default=10, help='Length of the trajectory sequence')
parser.add_argument('--dt', type=float, default=0.02, help='Time step size')
parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--n_steps', type=int, default=100, help='Number of steps per epoch')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for training')
parser.add_argument('--decay_step_size', type=int, default=10, help='Step size for learning rate decay')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for learning rate decay')
parser.add_argument('--rec_activation', type=str, default='relu', help='Recurrent activation function for the RNN')
parser.add_argument('--out_activation', type=str, default='softmax', help='Output activation function for the RNN')
parser.add_argument('--restore', type=str, default=None, help='Timestamp of the saved model to restore')
parser.add_argument('--preloaded_data', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use preloaded data')
parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to save the model')
parser.add_argument('--save_every', type=int, default=100, help='Save the model every n epochs')
parser.add_argument('--loss', type=str, default='CE', help='Loss function for training')
parser.add_argument('--is_wandb', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use wandb for logging')
parser.add_argument('--mode', type=str, default='train', help='Mode for running the model; input run folder name for model inspection')
parser.add_argument('--normalize_pc', type=str, default='softmax', help='Transformation applied to place cells in generation')
parser.add_argument('--truncating', type=int, default=0, help='Truncating steps for BPTT')
options = parser.parse_args()

if options.mode == 'train':
    # save directory
    now = time.strftime('%b-%d-%Y-%H-%M-%S', time.gmtime(time.time()))
    if options.restore is not None:
        now = options.restore
    options.save_dir = os.path.join('./results/rnn', now)

    if not os.path.exists(options.save_dir):
        os.makedirs(options.save_dir)
    print('Saving to:', options.save_dir)

    utils.save_options_to_json(options, os.path.join(options.save_dir, 'configs.json'))

    # define place cells, trajectory generator, model, and trainer
    place_cell = PlaceCells(options)
    generator = TrajectoryGenerator(options, place_cell)
    model = RNN(options, place_cell).to(options.device)
    trainer = Trainer(options, model, generator, place_cell, restore=options.restore)

    trainer.train(preloaded_data=options.preloaded_data, save=options.save)
    plot_place_cells(place_cell, options, res=30)
    plot_2d_performance(place_cell, generator, options, trainer)
    rate_map = compute_ratemaps(model, trainer, generator, options, res=20, n_avg=200, Ng=options.Ng)
    plot_2d_ratemaps(rate_map, options, n_col=4)
    plot_loss_err(trainer, options)

else:
    now = options.mode
    save_dir = os.path.join('./results/rnn', now)

    # load the configuration file to args
    t_args = argparse.Namespace()
    d = json.load(open(os.path.join(save_dir, 'configs.json')))
    for k in list(d.keys()):
        if k == '_get_args' or k == '_get_kwargs':
            del d[k]
    t_args.__dict__.update(d)
    options = parser.parse_args(namespace=t_args)
    print(options.__dict__)

    # load the model
    ckpt = torch.load(os.path.join(save_dir, 'models', 'most_recent_model.pth'))
    options.save_dir = save_dir
    place_cell = PlaceCells(options)
    model = RNN(options, place_cell).to(options.device)
    model.load_state_dict(ckpt)

    print('Plotting weights...')
    Wr = model.RNN.weight_hh_l0.detach().cpu().numpy()
    plot_weights(Wr, options)

    generator = TrajectoryGenerator(options, place_cell)
    trainer = Trainer(options, model, generator, place_cell, restore=False)
    print('Generating rate maps...')
    rate_map = compute_ratemaps(
        model, trainer, generator, options, res=30, n_avg=200, Ng=options.Ng
    )

    # calculate grid scores
    print('Generating low resolution rate maps...')
    lo_res = 20
    rate_map_lo_res = compute_ratemaps(
        model, trainer, generator, options, res=lo_res, n_avg=200, Ng=options.Ng
    )
    # scores are already sorted in descending order
    print('Calculating grid scores...')
    idx, scores = compute_grid_scores(lo_res, rate_map_lo_res, options) # descending order
    # select the top grid cells
    plot_all_ratemaps(rate_map[idx], options, scores)

    # save scores
    np.save(os.path.join(save_dir, 'grid_scores.npy'), scores)
    # save top 64 grid cells
    np.save(os.path.join(save_dir, 'top64_grid_cells.npy'), rate_map[idx[:64]])

    # border score
    print('Calculating border scores...')
    idx_border, scores_border = compute_border_scores(lo_res, rate_map_lo_res, options)
    plot_all_ratemaps(rate_map[idx_border], options, scores_border, dir='all_maps_border')