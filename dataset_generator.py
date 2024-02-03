import torch
import os
import numpy as np

from trajectory_generator import TrajectoryGenerator
from place_cells import PlaceCells

device = "cuda" if torch.cuda.is_available() else "cpu"

class Options:
    pass
options = Options()

options.batch_size = 50000      # number of trajectories per batch
options.sequence_length = 20  # number of steps in trajectory
options.learning_rate = 1e-4  # gradient descent learning rate
options.Np = 512              # number of place cells
options.Ng = 4096             # number of grid cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.DoG = True            # use difference of gaussians tuning curves
options.periodic = False      # trajectories with periodic boundary conditions
options.box_width = 2.2       # width of training environment
options.box_height = 2.2      # height of training environment
options.device = device       # use GPU if available

place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

gen = trajectory_generator.get_generator()
inputs, pc_outputs, pos = next(gen)

v = inputs[0]
init_actv = inputs[1]

# create a path called 'data' if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

np.savez('data/trajectory.npz',
    v=v.cpu().numpy(),
    init_actv=init_actv.cpu().numpy(),
    pc_outputs=pc_outputs.cpu().numpy(),
    pos=pos.cpu().numpy(),
    us=place_cells.us.cpu().numpy()
)

print(inputs[0].shape)  
print(inputs[1].shape)
print(pc_outputs.shape)
print(pos.shape)