# -*- coding: utf-8 -*-
import torch
import os
import numpy as np


class TrajectoryGenerator(object):

    def __init__(self, options, place_cells):
        self.options = options
        self.place_cells = place_cells

    def get_hidden_projector(self):
        g_r = torch.Generator()
        g_r.manual_seed(1)
        return torch.randn((self.options.Np, self.options.Ng), generator=g_r).to(self.options.device)

    def avoid_wall(self, position, hd, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:, 0]
        y = position[:, 1]
        dists = [box_width / 2 - x, box_height / 2 - y, box_width / 2 + x, box_height / 2 + y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(self, box_width, box_height, batch_size):
        '''Generate a random walk in a rectangular box'''
        samples = self.options.sequence_length
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias 
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples + 2, 2])
        head_dir = np.zeros([batch_size, samples + 2])
        position[:, 0, 0] = np.random.uniform(-box_width / 2, box_width / 2, batch_size)
        position[:, 0, 1] = np.random.uniform(-box_height / 2, box_height / 2, batch_size)
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        velocity = np.zeros([batch_size, samples + 2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])
        random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
        v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

        for t in range(samples + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            if not self.options.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:, t], head_dir[:, t], box_width, box_height)
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        # Periodic boundaries
        if self.options.periodic:
            position[:, :, 0] = np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
            position[:, :, 1] = np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        traj = {}
        # Input variables
        traj['init_hd'] = head_dir[:, 0, None]
        traj['init_x'] = position[:, 1, 0, None]
        traj['init_y'] = position[:, 1, 1, None]

        traj['ego_v'] = velocity[:, 1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

        # Target variables
        traj['target_hd'] = head_dir[:, 1:-1]
        traj['target_x'] = position[:, 2:, 0]
        traj['target_y'] = position[:, 2:, 1]

        return traj

    def get_generator(self, batch_size=None, box_width=None, box_height=None):
        '''
        Returns a generator that yields batches of trajectories
        '''
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        while True:
            traj = self.generate_trajectory(box_width, box_height, batch_size)

            v = np.stack([traj['ego_v'] * np.cos(traj['target_hd']),
                          traj['ego_v'] * np.sin(traj['target_hd'])], axis=-1)
            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

            pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
            # Put on GPU if GPU is available
            pos = pos.to(self.options.device)
            place_outputs = self.place_cells.get_activation(pos)

            init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to(self.options.device)
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            v = v.to(self.options.device)
            inputs = (v, init_actv)

            yield (inputs, place_outputs, pos)

    # def get_dataset(self, batch_size=None, box_width=None, box_height=None):
    #     """Returns a full dataset of trajectories
    #     """
    #     if not batch_size:
    #         batch_size = self.options.batch_size
    #     if not box_width:
    #         box_width = self.options.box_width
    #     if not box_height:
    #         box_height = self.options.box_height

        

    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):
        ''' For testing performance, returns a batch of smample trajectories'''
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        traj = self.generate_trajectory(box_width, box_height, batch_size)

        v = np.stack([traj['ego_v'] * np.cos(traj['target_hd']),
                      traj['ego_v'] * np.sin(traj['target_hd'])], axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.options.device)
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.options.device)
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        v = v.to(self.options.device)
        inputs = (v, init_actv)

        return (inputs, pos, place_outputs)


class TrajectoryGenerator1D(object):

    def __init__(self, options, place_cell):
        self.options = options
        self.place_cell = place_cell

    def avoid_wall(self, position):
        '''
        Avoid the wall by reflecting the position when the agent hits the wall

        Inputs:
            position: (batch_size) tensor of the CURRENT position of the agent
            dt: float, time step

        Outputs:
            position: (batch_size) tensor of the position of the agent after reflection
        '''
        track_length = self.place_cell.track_length
        dt = self.options.dt
        # check if the agent hits the wall
        hit_wall = (position.abs() >= track_length / 2)
        # if the agent hits the wall, current position should be border - the amount of overshoot
        sign = torch.sign(position[hit_wall])
        position[hit_wall] = sign * track_length / 2 - (position[hit_wall] - sign * track_length / 2)
        return position

    def generate_trajectory(self, seed=None):
        '''
        Generate a batch of trajectories

        Inputs:
            batch_size: int, number of trajectories to generate
            seq_len: int, length of each trajectory

        Outputs:
            traj: (batch_size, seq_len) tensor of the generated trajectories
        '''
        if seed:
            torch.manual_seed(seed)
        track_length = self.place_cell.track_length
        batch_size = self.options.batch_size
        seq_len = self.options.seq_len
        dt = self.options.dt
        # initialize position; add 2 to seq_len to account for initial and final positions
        # we will discard the final position later
        position = torch.zeros(batch_size, seq_len+2)

        # random initial position; between -track_length/2 and track_length/2
        position[:, 0] = torch.rand(batch_size) * track_length - track_length / 2

        # IMPORTANT: For nonperiodic boundary, using a positive velocity will like to result in bouncing back and forth near a wall
        velocity = torch.rand((batch_size, seq_len+1)) * 2 - 1

        for t in range(seq_len + 1):
            v = velocity[:, t]
            position[:, t+1] = position[:, t] + v * dt
            # for non-periodic boundary, check if the updated position hits the wall, if so reflect the position
            if not self.options.periodic:
                position[:, t+1] = self.avoid_wall(position[:, t+1])
        
        # for periodic boundary, wrap the position
        if self.options.periodic:
            position = (position + track_length / 2) % track_length - track_length / 2

        traj = {}
        traj['position'] = position[:, 1:-1].to(self.options.device) # discard initial and final positions, [batch_size, seq_len]
        traj['velocity'] = velocity[:, 1:].to(self.options.device) # discard initial velocity, [batch_size, seq_len]
        traj['init_position'] = position[:, 0].unsqueeze(-1).to(self.options.device)# [batch_size, 1]

        return traj

    def get_batch_data(self, seed=None):
        '''
        Generate a batch of trajectories and their corresponding place cell activations

        Outputs:
            input: (velocity, init_activation), where velocity is a (batch_size, seq_len, 1) tensor of the velocity of the agent
                and init_activation is a (batch_size, Np) tensor of the activation of the place cells at the initial position
            activation: (batch_size, seq_len, Np) tensor of the activation of the place cells
            position: (batch_size, seq_len, 1) tensor of the generated 1d trajectories
        '''
        traj = self.generate_trajectory(seed)
        # unsqueeze the position to match the shape of the place cell activations
        activation = self.place_cell.get_activation(traj['position'].unsqueeze(-1))
        init_activation = self.place_cell.get_activation(traj['init_position'].unsqueeze(-1))
        # unsqueeze the velocity to have 1 in the last dimension
        input = (traj['velocity'].unsqueeze(-1), init_activation.squeeze())
        return (input, activation, traj['position'].unsqueeze(-1))

    def get_generator(self, seed=None):
        while True:
            yield self.get_batch_data(seed)

    def get_test_batch(self, seed=None):
        return self.get_batch_data(seed)
