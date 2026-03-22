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
    "--inf_lr", type=float, default=[1e-2], nargs="+", help="Learning rate for inference"
)
parser.add_argument(
    "--n_module",
    type=int,
    default=1,
    help="Number of modules for block-wise inference LR expansion",
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
    "--restore", 
    type=str, 
    default=None, 
    help="Model path to load the weights from"
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
parser.add_argument("--place_cell_center_seed", type=int, default=0, help="Random seed for place cell center generation")
parser.add_argument("--lambda_z", type=float, default=0, help="Sparsity level for hidden activity")
parser.add_argument("--lambda_z_init", type=float, default=0, help="Sparsity level for initial hidden activity")
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
parser.add_argument(
    "--grid_n_shuffles",
    type=int,
    default=100,
    help="Number of shuffles for grid-cell significance thresholding",
)
parser.add_argument(
    "--grid_percentile",
    type=float,
    default=95.0,
    help="Percentile of shuffled null used as grid-cell threshold",
)
parser.add_argument(
    "--grid_min_shift_steps",
    type=int,
    default=1,
    help="Minimum circular shift (in sample steps) for shuffle null",
)

def expand_inf_lr(inf_lr, n_module, ng):
    if not isinstance(inf_lr, list):
        return inf_lr
    if len(inf_lr) == 1:
        return inf_lr[0]
    if len(inf_lr) == ng:
        return inf_lr
    if len(inf_lr) == n_module:
        if ng % n_module != 0:
            raise ValueError(
                f"Ng={ng} must be divisible by n_module={n_module} for block-wise inf_lr expansion."
            )
        block_size = ng // n_module
        return np.repeat(np.asarray(inf_lr, dtype=float), block_size).tolist()
    raise ValueError(
        f"inf_lr length must be 1, n_module ({n_module}), or Ng ({ng}); got {len(inf_lr)}."
    )

options = parser.parse_args()
options.inf_lr = expand_inf_lr(options.inf_lr, options.n_module, options.Ng)
options.periodic = PERIODIC
options.device = DEVICE
options.oned = ONED
options.weight_decay = WEIGHT_DECAY
options.decay_step_size = DECAY_STEP_SIZE
options.decay_rate = DECAY_RATE
# options.lambda_z = LAMBDA_Z
# options.lambda_z_init = LAMBDA_Z_INIT
options.surround_scale = SURROUND_SCALE

if options.mode == "train":
    # save directory
    now = time.strftime("%b-%d-%Y-%H-%M-%S", time.gmtime(time.time()))
    options.save_dir = os.path.join("./results/tpc", now)

    if options.save and not os.path.exists(options.save_dir):
        os.makedirs(options.save_dir)
        print("Saving to:", options.save_dir)
        utils.save_options_to_json(options, os.path.join(options.save_dir, "configs.json"))

    # define place cells, trajectory generator, model, and trainer
    place_cell = PlaceCells(options)
    plot_place_cells(place_cell, options, res=30)
    generator = TrajectoryGenerator(options, place_cell, environment=options.env_shape)
    model = TemporalPCN(options).to(options.device)
    init_model = HierarchicalPCN(options).to(options.device)
    trainer = PCTrainer(
        options, model, init_model, generator, place_cell
    )
    trainer.train(preloaded_data=options.preloaded_data, save=options.save)
    plot_2d_performance(place_cell, generator, options, trainer)
    res = 30
    rate_map = compute_ratemaps(
        model, trainer, generator, options, res=res, n_avg=200, Ng=options.Ng
    )
    plot_2d_ratemaps(rate_map, options, n_col=4, res=res)
    plot_loss_err(trainer, options)
    np.save(os.path.join(options.save_dir, "loss"), trainer.loss)

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
    options.inf_lr = expand_inf_lr(options.inf_lr, options.n_module, options.Ng)
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
        options, model, init_model, generator, place_cell
    )
    # Calculate grid scores on low-resolution maps (faster, standard for SAC scoring).
    lo_res = 20
    print("Generating low-resolution rate maps...")
    rate_map_lo_res = compute_ratemaps(
        model, trainer, generator, options, res=lo_res, n_avg=500, Ng=options.Ng
    )

    print("Calculating grid scores...")
    _, unsrt_scores, unsrt_sacs = compute_grid_scores(
        lo_res, rate_map_lo_res, options, srted=False
    )
    idx = np.flip(np.argsort(unsrt_scores))
    scores = [unsrt_scores[i] for i in idx]
    sacs = [unsrt_sacs[i] for i in idx]
    np.save(os.path.join(save_dir, "sac.npy"), sacs)

    print("Classifying significant grid cells with shuffled null...")
    grid_cls = classify_grid_cells_by_shuffled_null(
        model=model,
        trainer=trainer,
        trajectory_generator=generator,
        options=options,
        lo_res=lo_res,
        n_avg=500,
        Ng=options.Ng,
        n_shuffles=options.grid_n_shuffles,
        percentile=options.grid_percentile,
        min_shift_steps=options.grid_min_shift_steps,
    )
    is_grid = grid_cls["is_grid"]
    grid_threshold = grid_cls["threshold"]
    print(
        f"Grid threshold ({options.grid_percentile}th pct): {grid_threshold:.4f}. "
        f"Detected {is_grid.sum()}/{len(is_grid)} significant grid cells."
    )

    print("Generating high-resolution rate maps for visualization...")
    full_res = 50
    rate_map = compute_ratemaps(
        model, trainer, generator, options, res=full_res, n_avg=500, Ng=options.Ng
    )
    plot_all_ratemaps(rate_map[idx], options, full_res, scores)

    # grid scores for half of the fields
    field_height = rate_map_lo_res.shape[-1]
    _, left_scores, left_sacs = compute_grid_scores(
        lo_res, rate_map_lo_res[:, :, :field_height // 2], options, half=True, srted=False
    )
    _, right_scores, right_sacs = compute_grid_scores(
        lo_res, rate_map_lo_res[:, :, field_height // 2:], options, half=True, srted=False
    )

    # save scores
    np.save(os.path.join(save_dir, "grid_scores.npy"), scores)
    np.save(os.path.join(save_dir, "unsrt_grid_scores.npy"), unsrt_scores)
    np.save(os.path.join(save_dir, "left_grid_scores.npy"), left_scores)
    np.save(os.path.join(save_dir, "right_grid_scores.npy"), right_scores)
    np.save(os.path.join(save_dir, "left_sac.npy"), left_sacs)
    np.save(os.path.join(save_dir, "right_sac.npy"), right_sacs)
    np.save(os.path.join(save_dir, "is_grid.npy"), is_grid)
    np.save(os.path.join(save_dir, "grid_score_threshold.npy"), np.array([grid_threshold]))
    # save full map set sorted by descending grid score
    np.save(os.path.join(save_dir, "grid_maps.npy"), rate_map[idx])


    # border score
    # print("Calculating border scores...")
    # idx_border, scores_border = compute_border_scores(lo_res, rate_map_lo_res, options)
    # plot_all_ratemaps(
    #     rate_map[idx_border], options, scores_border, dir="all_maps_border"
    # )
