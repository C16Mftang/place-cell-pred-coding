# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

from visualize import save_ratemaps
import os
from tqdm import tqdm

from get_data import *
import utils

class Trainer(object):
    def __init__(self, options, model, trajectory_generator, place_cells, restore=True):
        self.options = options
        self.model = model
        self.place_cells = place_cells
        self.trajectory_generator = trajectory_generator
        self.lr = self.options.learning_rate
        self.n_epochs = self.options.n_epochs
        self.n_steps = self.options.n_steps

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=options.decay_step_size, 
            gamma=options.decay_rate,
        )

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, 'models')
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")

    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].
        '''
        inputs = (inputs[0].to(self.options.device), inputs[1].to(self.options.device))
        pc_outputs = pc_outputs.to(self.options.device)
        pos = pos.to(self.options.device)
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self, preloaded_data=None, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            preloaded_data: If true, load pre-generated data from file.
            save: If true, save a checkpoint after each epoch.
        '''
        if not preloaded_data:
            # Construct generator
            gen = self.trajectory_generator.get_generator()
        else:
            dt = str(self.options.dt).replace('.','')
            dpath = 'data/trajectory1d' if self.options.oned else 'data/trajectory'
            path = os.path.join(
                dpath, 
                f'{self.options.batch_size*self.options.n_steps}_{self.options.sequence_length}_{self.options.Np}_{dt}.npz'
            )
            # check if the file exists
            if os.path.exists(path):
                print(f'Loading pre-generated data at {path}...')
            else:
                print(f'Generating new data at {path}...')
                generate_traj_data(self.options)

            dataloader = get_traj_loader(path, self.options)

        for epoch_idx in range(self.n_epochs):
            epoch_loss = 0
            epoch_err = 0

            iterable = dataloader if preloaded_data else range(self.n_steps)
            tbar = tqdm(iterable, leave=True)

            for item in tbar:
                if preloaded_data:
                    inputs, pc_outputs, pos = item
                else:
                    inputs, pc_outputs, pos = next(gen)

                loss, err = self.train_step(inputs, pc_outputs, pos)
                epoch_loss += loss
                epoch_err += err

                tbar.set_description(
                    'Epoch: {}/{}. Loss: {}. Err: {} cm.'.format(
                        epoch_idx+1, self.n_epochs,
                        np.round(loss, 4), 
                        np.round(100 * err, 2),
                    )
                )
                
            self.loss.append(epoch_loss / self.n_steps)
            self.err.append(epoch_err / self.n_steps)

            # Update learning rate
            self.scheduler.step()

            if save and (epoch_idx + 1) % self.options.save_every == 0:
                # Save checkpoint
                ckpt_path = os.path.join(
                    self.ckpt_dir, 
                    f'epoch_{epoch_idx}.pth'
                )
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(
                        self.options.ckpt_dir,
                        'most_recent_model.pth'
                    )
                )
        tbar.close()

    def predict(self, inputs):
        pred_pos = self.model.predict(inputs)
        return pred_pos, None

class PCTrainer(object):
    def __init__(self, options, model, init_model, trajectory_generator, place_cells, restore=True):
        self.options = options
        self.model = model
        self.init_model = init_model
        self.trajectory_generator = trajectory_generator
        self.place_cells = place_cells
        self.lr = options.learning_rate
        self.inf_iters = options.inf_iters
        self.test_inf_iters = options.test_inf_iters
        self.inf_lr = options.inf_lr
        self.n_epochs = options.n_epochs
        self.n_steps = options.n_steps

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
        )
        self.init_optimizer = torch.optim.Adam(
            self.init_model.parameters(),
            lr=self.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=options.decay_step_size, 
            gamma=options.decay_rate,
        )

        self.loss = []
        self.err = []
        self.energy = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, 'models')
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")

    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].
        '''
        self.model.train()
        self.init_model.train()
        total_loss = 0 # average loss across time steps
        total_energy = 0 # average energy across time steps
        vs, init_actv = inputs[0].to(self.options.device), inputs[1].to(self.options.device)

        # train the initial static pcn to get the initial hidden state
        self.init_optimizer.zero_grad()
        self.init_model.inference(self.inf_iters, self.inf_lr, init_actv)
        energy, obs_loss = self.init_model.get_energy()
        energy.backward()
        self.init_optimizer.step()

        total_loss += obs_loss.item()
        total_energy += energy.item()

        # get the initial hidden state from the initial static model
        prev_hidden = self.init_model.z.clone().detach()
        for k in range(self.options.sequence_length):
            p = pc_outputs[:, k].to(self.options.device)
            v = vs[:, k].to(self.options.device)
            self.optimizer.zero_grad()
            self.model.inference(self.inf_iters, self.inf_lr, v, prev_hidden, p)
            energy, obs_loss = self.model.get_energy()
            energy.backward()
            self.optimizer.step()

            # update the hidden state
            prev_hidden = self.model.z.clone().detach()

            # add up the loss value at each time step
            total_loss += obs_loss.item()
            total_energy += energy.item()

        return total_energy / (self.options.sequence_length + 1), total_loss / (self.options.sequence_length + 1)

    def train(self, preloaded_data=None, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            preloaded_data: If true, load pre-generated data from file.
            save: If true, save a checkpoint after each epoch.
        '''

        if not preloaded_data:
            # Construct generator
            gen = self.trajectory_generator.get_generator()
        else:
            dt = str(self.options.dt).replace('.','')
            dpath = 'data/trajectory1d' if self.options.oned else 'data/trajectory'
            path = os.path.join(
                dpath, 
                f'{self.options.batch_size*self.options.n_steps}_{self.options.sequence_length}_{self.options.Np}_{dt}.npz'
            )
            # check if the file exists
            if os.path.exists(path):
                print(f'Loading pre-generated data at {path}...')
            else:
                print(f'Generating new data at {path}...')
                generate_traj_data(self.options)

            dataloader = get_traj_loader(path, self.options)

        for epoch_idx in range(self.n_epochs):
            epoch_loss = 0
            epoch_energy = 0
            epoch_err = 0

            iterable = dataloader if preloaded_data else range(self.n_steps)
            tbar = tqdm(iterable, leave=True)

            for item in tbar:
                if preloaded_data:
                    inputs, pc_outputs, pos = item
                else:
                    inputs, pc_outputs, pos = next(gen)
            
                energy, loss = self.train_step(inputs, pc_outputs, pos)
                pred_xs, _ = self.predict(inputs)
                if not isinstance(self.options.out_activation, utils.Softmax):
                    pred_xs = F.softmax(pred_xs, dim=-1)

                pred_pos = self.place_cells.get_nearest_cell_pos(pred_xs)
                err = torch.sqrt(((pos.to(self.options.device) - pred_pos)**2).sum(-1)).mean().item()
                epoch_err += err
                epoch_loss += loss
                epoch_energy += energy

                tbar.set_description(
                    'Epoch: {}/{}. Loss: {}. PC Energy: {}. Err: {} cm.'.format(
                        epoch_idx+1, self.n_epochs,
                        np.round(loss, 4), 
                        np.round(energy, 4),
                        np.round(100 * err, 2),
                    )
                )

            self.loss.append(epoch_loss / self.n_steps)
            self.err.append(epoch_err / self.n_steps)
            self.energy.append(epoch_energy / self.n_steps)

            # Update learning rate
            self.scheduler.step()

            if save and (epoch_idx + 1) % self.options.save_every == 0:
                # Save checkpoint
                ckpt_path = os.path.join(
                    self.ckpt_dir, 
                    f'epoch_{epoch_idx}.pth'
                )
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(
                        self.options.ckpt_dir,
                        'most_recent_model.pth'
                    )
                )

        tbar.close()

    def predict(self, inputs):
        self.model.eval()
        self.init_model.eval()
        vs, init_actv = inputs
        pred_zs = []
        with torch.no_grad():
            self.init_model.inference(self.test_inf_iters, self.inf_lr, init_actv.to(self.options.device))
            prev_hidden = self.init_model.z.clone().detach()
            for k in range(self.options.sequence_length):
                v = vs[:, k].to(self.options.device)
                prev_hidden = self.model.g(v, prev_hidden)
                pred_zs.append(prev_hidden)

            pred_zs = torch.stack(pred_zs, dim=1) # [batch_size, sequence_length, Ng]
            pred_xs = self.model.decode(pred_zs)
            
        return pred_xs, pred_zs