from torch.utils.data import Dataset
import numpy as np
import torch
import os

from trajectory_generator import TrajectoryGenerator
from place_cells import PlaceCells

def generate_traj_data(options):
    # generate a batch of full size data and save it to a .npz file
    options.batch_size = options.full_size
    place_cells = PlaceCells(options)
    trajectory_generator = TrajectoryGenerator(options, place_cells)

    gen = trajectory_generator.get_generator()
    inputs, pc_outputs, pos = next(gen)

    v = inputs[0]
    init_actv = inputs[1]

    # create a path called 'data' if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    np.savez(f'data/trajectory_{options.full_size}_{options.sequence_length}_{options.Np}.npz',
        v=v.cpu().numpy(),
        init_actv=init_actv.cpu().numpy(),
        pc_outputs=pc_outputs.cpu().numpy(),
        pos=pos.cpu().numpy(),
        us=place_cells.us.cpu().numpy()
    )

    print(f'Generated {options.full_size} trajectories. Sizes: velocity: {v.shape}, \
        initial activation: {init_actv.shape}, \
        place cell outputs: {pc_outputs.shape}, \
        position: {pos.shape}'
    )

class Trjectory(Dataset):
    def __init__(self, npz_path):
        # Load the .npz file
        self.data = np.load(npz_path, mmap_mode='r')

        # Access the individual arrays. The keys ('v', 'init_activation', etc.)
        # need to match the names used when the .npz file was created.
        self.v = self.data['v']  # Shape: [20, 50000, 2]
        self.init_actv = self.data['init_actv']  # Shape: [50000, 512]
        self.pc_outputs = self.data['pc_outputs']  # Shape: [20, 50000, 512]
        self.pos = self.data['pos']  # Shape: [20, 50000, 2]

    def __len__(self):
        # Assuming all arrays have the same number of samples (50000)
        return self.v.shape[1]

    def __getitem__(self, idx):
        # Fetch the elements for the given index
        v_sample = self.v[:, idx, :]  # Shape: [20, 2]
        init_activation_sample = self.init_actv[idx, :]  # Shape: [512]
        pc_outputs_sample = self.pc_outputs[:, idx, :]  # Shape: [20, 512]
        pos_sample = self.pos[:, idx, :]  # Shape: [20, 2]

        # Inputs is a tuple of (v, init_activation)
        inputs = (v_sample, init_activation_sample)

        # Return a tuple of inputs, pc_output, and pos
        return inputs, pc_outputs_sample, pos_sample

def get_traj_loader(path, options):
    # Create a Trajectory dataset
    dataset = Trjectory(path)

    # Create a DataLoader from the Trajectory dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    return loader

