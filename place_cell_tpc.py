import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
import argparse
import json

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
parser.add_argument("--dt", type=float, default=0.02, help="Time step size")
parser.add_argument(
    "--batch_size", type=int, default=500, help="Batch size for training"
)
parser.add_argument(
    "--n_epochs", type=int, default=150, help="Number of training epochs"
)
parser.add_argument(
    "--n_steps", type=int, default=100, help="Number of steps in each trajectory"
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
parser.add_argument(
    "--fixed_rf", 
    type=lambda x: (str(x).lower() == "true"), 
    default=True, 
    help="Use fixed place cell receptive fields"
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
options.place_cell_rf = PLACE_CELL_RF
options.surround_scale = SURROUND_SCALE

if options.mode == "train":
    # save directory
    now = time.strftime("%b-%d-%Y-%H-%M-%S", time.gmtime(time.time()))
    if options.restore is not None:
        now = options.restore
    options.save_dir = os.path.join("./results/tpc", now)

    if options.save and not os.path.exists(options.save_dir):
        os.makedirs(options.save_dir)
        print("Saving to:", options.save_dir)
        utils.save_options_to_json(options, os.path.join(options.save_dir, "configs.json"))

    # define place cells, trajectory generator, model, and trainer
    place_cell = PlaceCells(options)
    generator = TrajectoryGenerator(options, place_cell, environment=options.env_shape)
    model = TemporalPCN(options).to(options.device)
    init_model = HierarchicalPCN(options).to(options.device)
    trainer = PCTrainer(
        options, model, init_model, generator, place_cell, restore=options.restore
    )
    trainer.train(preloaded_data=options.preloaded_data, save=options.save)
    plot_place_cells(place_cell, options, res=30)
    plot_2d_performance(place_cell, generator, options, trainer)
    res = 30
    rate_map = compute_ratemaps(
        model, trainer, generator, options, res=res, n_avg=200, Ng=options.Ng
    )
    plot_2d_ratemaps(rate_map, options, n_col=4, res=res)
    plot_loss_err(trainer, options)

else:
    now = options.mode
    save_dir = os.path.join("./results/tpc", now)

    # load the configuration file to args
    t_args = argparse.Namespace()
    d = json.load(open(os.path.join(save_dir, "configs.json")))
    for k in list(d.keys()):
        if k == "_get_args" or k == "_get_kwargs":
            del d[k]
    t_args.__dict__.update(d)
    options = parser.parse_args(namespace=t_args)
    print(options.__dict__)

    # load the model
    ckpt = torch.load(os.path.join(save_dir, "models", "most_recent_model.pth"))
    options.save_dir = save_dir
    model = TemporalPCN(options).to(options.device)
    init_model = HierarchicalPCN(options).to(options.device)
    model.load_state_dict(ckpt["model"])
    init_model.load_state_dict(ckpt["init_model"])

    print("Plotting weights...")
    Wr = model.Wr.weight.detach().cpu().numpy()
    plot_weights(Wr, options)

    place_cell = PlaceCells(options)
    generator = TrajectoryGenerator(options, place_cell, environment=options.env_shape)
    trainer = PCTrainer(
        options, model, init_model, generator, place_cell, restore=False
    )
    print("Generating rate maps...")
    full_res = 30
    rate_map = compute_ratemaps(
        model, trainer, generator, options, res=full_res, n_avg=200, Ng=options.Ng
    )

    # calculate grid scores
    print("Generating low resolution rate maps...")
    lo_res = 20
    rate_map_lo_res = compute_ratemaps(
        model, trainer, generator, options, res=lo_res, n_avg=200, Ng=options.Ng
    )
    # scores are already sorted in descending order
    print("Calculating grid scores...")
    idx, scores, sacs = compute_grid_scores(lo_res, rate_map_lo_res, options)  # descending order
    # select the top grid cells
    plot_all_ratemaps(rate_map[idx], options, full_res, scores)
    np.save(os.path.join(save_dir, "sac.npy"), sacs)

    # grid scores for half of the fields
    field_height = rate_map_lo_res.shape[-1]
    _, left_scores, _ = compute_grid_scores(
        lo_res, rate_map_lo_res[:, :, :field_height // 2], options, half=True
    )
    _, right_scores, _ = compute_grid_scores(
        lo_res, rate_map_lo_res[:, :, field_height // 2:], options, half=True
    )

    # save scores
    np.save(os.path.join(save_dir, "grid_scores.npy"), scores)
    np.save(os.path.join(save_dir, "left_grid_scores.npy"), left_scores)
    np.save(os.path.join(save_dir, "right_grid_scores.npy"), right_scores)
    # save top 64 grid cells
    np.save(os.path.join(save_dir, "top64_grid_cells.npy"), rate_map[idx[:64]])

    # border score
    # print("Calculating border scores...")
    # idx_border, scores_border = compute_border_scores(lo_res, rate_map_lo_res, options)
    # plot_all_ratemaps(
    #     rate_map[idx_border], options, scores_border, dir="all_maps_border"
    # )
