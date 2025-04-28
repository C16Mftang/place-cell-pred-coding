import numpy as np
import torch
import os
from matplotlib import pyplot as plt
import argparse
import time
import json
import wandb
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

if options.mode == 'train':
    if not options.sweep:
        now = time.strftime("%b-%d-%Y-%H-%M-%S", time.gmtime(time.time()))
        options.save_dir = os.path.join("./results/pcn", now)
        if not os.path.exists(options.save_dir):
            os.makedirs(options.save_dir)
        if not os.path.exists(os.path.join(options.save_dir, "models")):
            os.makedirs(os.path.join(options.save_dir, "models"))
        print("Saving to:", options.save_dir)
        utils.save_options_to_json(options, os.path.join(options.save_dir, "configs.json"))

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

        if (epoch + 1) % options.save_every == 0 and not options.sweep:
            torch.save(
                pcn.state_dict(),
                os.path.join(options.save_dir, "models", f"model{epoch + 1}.pth"),
            )
    
    if not options.sweep:
        # Plot the training loss and save
        np.save(os.path.join(options.save_dir, "training_loss.npy"), np.array(train_mses))
        plt.plot(train_mses)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title("Training Loss")
        plt.savefig(os.path.join(options.save_dir, "training_loss.png"))
        plt.close()

    if options.is_wandb:
        wandb.finish()

else:
    now = options.mode
    save_dir = os.path.join("./results/pcn", now)

    # load the configuration file to args
    t_args = argparse.Namespace()
    d = json.load(open(os.path.join(save_dir, "configs.json")))
    for k in list(d.keys()):
        if k == "_get_args" or k == "_get_kwargs":
            del d[k]
    t_args.__dict__.update(d)
    options = parser.parse_args(namespace=t_args)
    print(options.__dict__)

    place_cells = PlaceCells(options)
    trajectory_generator = TrajectoryGenerator(options, place_cells)

    # load the model
    nodes = [options.Ng, options.Np]
    pcn = MultilayerPCN(
        nodes,
        options.out_activation,
        lamb=options.lambda_z_init,
        use_bias=False,
        relu_inf=options.relu_inf,
    ).to(device)

    ckpt = torch.load(os.path.join(save_dir, "models", f"model{options.n_epochs}.pth"))
    options.save_dir = save_dir
    pcn.load_state_dict(ckpt)

    pcn.set_sparsity(0.0)
    options.sequence_length = 1
    options.dt = 0.01
    n_avg = 200
    g = np.zeros([n_avg, options.batch_size * options.sequence_length, options.Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])
    activations = np.zeros([options.Ng, options.res, options.res])
    counts = np.zeros([options.res, options.res])
    for index in tqdm(range(n_avg)):
        # pos_batch: [batch_size, sequence_length, 2]
        _, _, pos_batch = trajectory_generator.get_test_batch()

        pc_outputs_test = place_cells.get_activation(pos_batch).detach()
        pc_outputs_test = pc_outputs_test.reshape((-1, options.Np))
        pc_outputs_test = pc_outputs_test - pc_outputs_test.mean(dim=0, keepdim=True)
        pcn.inference(
            pc_outputs_test, options.inf_iters_test, options.inf_lr_test
        )
        g_batch = pcn.val_nodes[0].clone().detach().cpu().numpy()  # [Ng, res**2]
        pos_batch = np.reshape(pos_batch.cpu().detach().numpy(), [-1, 2])

        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (
            (pos_batch[:, 0] + options.box_width / 2) / (options.box_width) * options.res
        )
        y_batch = (
            (pos_batch[:, 1] + options.box_height / 2) / (options.box_height) * options.res
        )

        for i in range(options.batch_size * options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >= 0 and x < options.res and y >= 0 and y < options.res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    # make it a density map
    for x in range(options.res):
        for y in range(options.res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    g = g.reshape([-1, options.Ng])
    pos = pos.reshape([-1, 2])

    idx, scores, sacs = compute_grid_scores(options.res, activations, options)
    sorted_gcs = activations[idx]
    plot_all_ratemaps(sorted_gcs, options, scores=scores)
    np.save(os.path.join(save_dir, "grid_scores.npy"), scores)
    np.save(os.path.join(save_dir, "grid_maps.npy"), sorted_gcs)

