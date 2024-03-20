from torch.utils.data import Dataset
import numpy as np
import torch
import os

from trajectory_generator import TrajectoryGenerator, TrajectoryGenerator1D
from place_cells import PlaceCells, PlaceCells1D

def generate_traj_data(options):
    # generate a batch of full size data and save it to a .npz file
    # convert the batch size, which is the input to the generator, to the full size
    bsz = options.batch_size
    options.batch_size = bsz * options.n_steps
    if options.oned:
        place_cells = PlaceCells1D(options)
        trajectory_generator = TrajectoryGenerator1D(options, place_cells)
    else:
        place_cells = PlaceCells(options)
        trajectory_generator = TrajectoryGenerator(options, place_cells)

    gen = trajectory_generator.get_generator()
    inputs, pc_outputs, pos = next(gen)

    v = inputs[0]
    init_actv = inputs[1]

    dt = str(options.dt).replace('.','')
    dpath = 'data/trajectory1d' if options.oned else 'data/trajectory'
    if not os.path.exists(dpath):
        os.makedirs(dpath)
        
    np.savez(
        os.path.join(dpath, f'{options.batch_size}_{options.sequence_length}_{options.Np}_{dt}.npz'),
        v=v.cpu().numpy(),
        init_actv=init_actv.cpu().numpy(),
        pc_outputs=pc_outputs.cpu().numpy(),
        pos=pos.cpu().numpy(),
        centers=place_cells.centers.cpu().numpy()
    )

    print(f'Generated {options.batch_size} trajectories. Sizes:') 
    print(f'velocity: {v.shape}')
    print(f'initial activation: {init_actv.shape}')
    print(f'place cell outputs: {pc_outputs.shape}')
    print(f'position: {pos.shape}')

    # reset the batch size
    options.batch_size = bsz

class Trjectory(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        # Load the .npz file
        self.data = np.load(npz_path, mmap_mode='r')

        # Access the individual arrays. The keys ('v', 'init_activation', etc.)
        # need to match the names used when the .npz file was created.
        self.v = self.data['v']  # Shape: [batch_size, sequence_length, 1]
        self.init_actv = self.data['init_actv']  # Shape: [batch_size, Np]
        self.pc_outputs = self.data['pc_outputs']  # Shape: [batch_size, sequence_length, Np]
        self.pos = self.data['pos']  # Shape: [batch_size, sequence_length, 1]

    def __len__(self):
        # Assuming all arrays have the same number of samples (50000)
        return self.v.shape[0]

    def __getitem__(self, idx):
        # Fetch the elements for the given index
        v_sample = self.v[idx]  # Shape: [sequence_length, 1]
        init_activation_sample = self.init_actv[idx]  # Shape: [Np]
        pc_outputs_sample = self.pc_outputs[idx]  # Shape: [sequence_length, Np]
        pos_sample = self.pos[idx]  # Shape: [sequence_length, 1]

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

