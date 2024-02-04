import numpy as np
import torch
import os
from matplotlib import pyplot as plt

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_grid_cells(gcs, epoch):
    # visualize latent space
    n_show = 10
    # set seed
    np.random.seed(0)
    select = np.random.choice(gcs.shape[1], n_show**2, replace=False)
    # sort the select
    select = np.sort(select)
    gcs = gcs[:, select]
    fig, axes = plt.subplots(n_show, n_show, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        gc = gcs[:,i].real.reshape((res, res))
        gc = (gc - np.min(gc)) / (np.max(gc) - np.min(gc) + 1e-6)
        im = ax.imshow(gc, cmap='jet')
        ax.set_title(f'latent {select[i]}', fontsize=8)
        ax.axis('off')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(options.save_dir + f'latent_space_epoch{epoch}.png')
    plt.close()

# Training options and hyperparameters
class Options:
    pass
options = Options()

options.save_dir = './models/pcn/'
options.Np = 1000              # number of place cells
options.Ng = 100             # number of grid cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.DoG = True      # use difference of gaussians tuning curves
options.periodic = True      # trajectories with periodic boundary conditions
options.box_width = 1.6       # width of training environment
options.box_height = 1.6      # height of training environment
options.device = device       # specify devices

if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)

# predictive coding hyperparameters
res = 30
options.nonlin = 'Tanh'  
options.inference_lr = 1e-2
options.inference_iters = 20
options.learning_lr = 2e-3
options.learning_iters = 600
options.batch_size = 100
options.sample_size = res ** 2
options.decay_step_size = 50
options.decay_rate = 0.5
options.weight_decay = 1e-5
options.lambda_hidden = 0.1

place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

# Generate positions on a 30x30 grid
# each position can then be represented as Np-dimensional vector by place cells
pos = np.array(np.meshgrid(np.linspace(-options.box_width/2, options.box_width/2, res),
                         np.linspace(-options.box_height/2, options.box_height/2, res))).T
pos = torch.tensor(pos).to(device)

# get place cell activations
pc_outputs = place_cells.get_activation(pos).detach().cpu()
pc_outputs = pc_outputs.reshape(res * res, options.Np)
pc_outputs = pc_outputs - pc_outputs.mean(dim=0, keepdim=True)

# Train the PCN
nodes = [options.Ng, options.Np]
pcn = HierarchicalPCN(nodes, options.nonlin, lamb=options.lambda_hidden, use_bias=False).to(device)
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
for i in range(options.learning_iters):
    iter_loss = 0
    for batch_idx in range(0, options.sample_size, options.batch_size):
        data = X[batch_idx:batch_idx+options.batch_size]
        optimizer.zero_grad()
        pcn.inference(data, options.inference_iters, options.inference_lr)
        loss = pcn.get_energy()
        loss.backward()
        optimizer.step()

        iter_loss += loss.item() / (options.sample_size / options.batch_size)

    if i % 10 == 0:
        print('Epoch', i)
        print('Loss', iter_loss)

    # run inference every 50 epochs
    if i == 0 or (i + 1) % (options.learning_iters // 10) == 0:
        pcn.lamb = 0.
        pcn.inference(X, options.inference_iters, inf_lr=1e-3)
        gcs = pcn.val_nodes[0].clone().detach().cpu().numpy()
        visualize_grid_cells(gcs, i)
        pcn.lamb = options.lambda_hidden

    train_mse = iter_loss
    train_mses.append(train_mse)
    scheduler.step()

# Plot the training loss and save
plt.plot(train_mses)
plt.xlabel('Epoch')
plt.ylabel('Energy')
plt.title('Training Loss')
plt.savefig(options.save_dir + 'training_loss.png')
plt.close()

# simply run PCA on the place cell activations
pc_cov = np.cov(pc_outputs.cpu().numpy())
# eigendecomposition of covariance matrix
eigvals, eigvecs = np.linalg.eig(pc_cov)
# sort by eigenvalues
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]

# visualize the eigenvecs
n_show = 10
fig, axes = plt.subplots(n_show, n_show, figsize=(10,10))
for i, ax in enumerate(axes.flat):
    gc = eigvecs[:,i].real.reshape((res, res))
    gc = (gc - np.min(gc)) / (np.max(gc) - np.min(gc))
    im = ax.imshow(gc, cmap='jet')
    ax.set_title(f'eivector {i}', fontsize=8)
    ax.axis('off')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig(options.save_dir + 'eigenvectors.png')
plt.close()
