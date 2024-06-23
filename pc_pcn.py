import numpy as np
import torch
import os
from matplotlib import pyplot as plt

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import *
from visualize import compute_grid_scores
import argparse
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_grid_cells(gcs, epoch, options):
    # visualize latent space
    n_show = 10
    # set seed
    np.random.seed(0)
    select = np.random.choice(gcs.shape[0], n_show**2, replace=False)
    # sort the select
    select = np.sort(select)
    gcs = gcs[select]
    fig, axes = plt.subplots(n_show, n_show, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        gc = gcs[i].real.reshape((options.res, options.res))
        gc = (gc - np.min(gc)) / (np.max(gc) - np.min(gc) + 1e-6)
        im = ax.imshow(gc, cmap='jet')
        ax.set_title(f'latent {select[i]}', fontsize=8)
        ax.axis('off')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(options.save_dir + f'latent_space_epoch{epoch}.png')
    plt.close()

def plot_all_ratemaps(rate_map, scores, options):
    n_col = 8
    Ng_per_file = 64
    n_files = options.Ng // Ng_per_file

    all_dir = os.path.join(options.save_dir, 'all_maps')
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)

    for i in tqdm(range(n_files)):
        rm = rate_map[i*Ng_per_file:(i+1)*Ng_per_file] # [Ng_per_file, res, res]
        fig, axes = plt.subplots(Ng_per_file//n_col, n_col, figsize=(n_col, Ng_per_file//n_col))
        for j, ax in enumerate(axes.flatten()):
            r = (rm[j] - rm[j].min()) / (rm[j].max() - rm[j].min() + 1e-9)
            ax.imshow(r, cmap='jet')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{scores[j+i*Ng_per_file]:.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(all_dir, f'2d_ratemaps_{i}.png'))
        plt.close(fig)

# Training options and hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--save_dir', type=str, default='./results/pcn/', help='Directory to save the models')
parser.add_argument('--Np', type=int, default=512, help='Number of place cells')
parser.add_argument('--Ng', type=int, default=256, help='Number of grid cells')
parser.add_argument('--place_cell_rf', type=float, default=0.12, help='Width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale', type=int, default=2, help='If DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--DoG', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use difference of gaussians tuning curves')
parser.add_argument('--periodic', type=lambda x: (str(x).lower() == 'true'), default=False, help='Trajectories with periodic boundary conditions')
parser.add_argument('--box_width', type=float, default=1.4, help='Width of training environment')
parser.add_argument('--box_height', type=float, default=1.4, help='Height of training environment')
parser.add_argument('--out_activation', type=str, default='linear', help='Nonlinearity function')
parser.add_argument('--inference_lr', type=float, default=1e-2, help='Learning rate for inference')
parser.add_argument('--inference_lr_test', type=float, default=1e-3, help='Learning rate for inference during testing')
parser.add_argument('--inference_iters', type=int, default=20, help='Number of inference iterations')
parser.add_argument('--learning_lr', type=float, default=2e-3, help='Learning rate for training')
parser.add_argument('--learning_iters', type=int, default=600, help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--res', type=int, default=30, help='Sample resolution')
parser.add_argument('--decay_step_size', type=int, default=10, help='Step size for learning rate decay')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
parser.add_argument('--lambda_z_init', type=float, default=0.1, help='Lambda for hidden layer')
parser.add_argument('--normalize_pc', type=str, default='softmax', help='transformation applied to place cells in generation')
parser.add_argument('--relu_inf', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use ReLU for inference')

options = parser.parse_args()

options.save_dir = os.path.join(options.save_dir, f'sparse_{(options.lambda_z_init != 0)}_relu_{options.relu_inf}')
if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)

place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

# Generate positions on a 30x30 grid
# each position can then be represented as Np-dimensional vector by place cells
# this provides a 30x30 approximation of the whole environment
pos = np.array(
    np.meshgrid(
        np.linspace(-options.box_width/2, options.box_width/2, options.res),
        np.linspace(-options.box_height/2, options.box_height/2, options.res)
    )
).T
pos = torch.tensor(pos).to(device)

# get place cell activations
pc_outputs = place_cells.get_activation(pos).detach().cpu()
pc_outputs = pc_outputs.reshape(options.res ** 2, options.Np)
pc_outputs = pc_outputs - pc_outputs.mean(dim=0, keepdim=True)

# Train the PCN
nodes = [options.Ng, options.Np]
pcn = MultilayerPCN(nodes, options.out_activation, lamb=options.lambda_z_init, use_bias=False, relu_inf=options.relu_inf).to(device)
X = torch.tensor(pc_outputs).to(device)
optimizer = torch.optim.Adam(pcn.parameters(), 
    lr=options.learning_lr, 
    weight_decay=options.weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
    step_size=options.decay_step_size, 
    gamma=options.decay_rate
)

train_mses = []
sample_size = int(options.res**2)
for i in range(options.learning_iters):
    iter_loss = 0
    tbar = tqdm(range(0, sample_size, options.batch_size))
    for batch_idx in tbar:
        data = X[batch_idx:batch_idx+options.batch_size]
        optimizer.zero_grad()
        pcn.inference(data, options.inference_iters, options.inference_lr)
        loss = pcn.get_energy()
        loss.backward()
        optimizer.step()
        iter_loss += loss.item() 

        tbar.set_description(
            'Epoch: {}/{}. Loss: {}'.format(
                i + 1, 
                options.learning_iters,
                np.round(loss.item(), 4), 
            )
        )

    train_mse = iter_loss / (sample_size / options.batch_size)
    train_mses.append(train_mse)
    scheduler.step()

# Plot the training loss and save
plt.plot(train_mses)
plt.xlabel('Epoch')
plt.ylabel('Energy')
plt.title('Training Loss')
plt.savefig(os.path.join(options.save_dir, 'training_loss.png'))
plt.close()

# save model
# torch.save(pcn.state_dict(), options.save_dir + 'pcn.pth')

pcn.set_sparsity(0.)
pcn.inference(X, options.inference_iters, options.inference_lr_test)
gcs = pcn.val_nodes[0].clone().detach().cpu().numpy().T # [Ng, res**2]
# visualize_grid_cells(gcs, options.learning_iters, options)
gcs = gcs.reshape((-1, options.res, options.res))
idx, scores = compute_grid_scores(options.res, gcs, options)
sorted_gcs = gcs[idx]
plot_all_ratemaps(sorted_gcs, scores, options)
np.savez(os.path.join(options.save_dir, 'gc_and_scores.npz'), gcs=sorted_gcs, scores=scores)


