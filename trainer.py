# -*- coding: utf-8 -*-
import torch
import numpy as np

from visualize import save_ratemaps
import os


class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=True):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=lr,
            # weight_decay=options.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
            step_size=options.decay_step_size, 
            gamma=options.decay_rate,
        )

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train_batch(self, loader, n_epochs: int = 1000, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_epochs: Number of training epochs
            loader: DataLoader object
        '''

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(n_epochs):
            for batch_idx, (inputs, pc_outputs, pos) in enumerate(loader):
                inputs = (inputs[0].transpose(0, 1).to(self.options.device), 
                    inputs[1].to(self.options.device))
                pc_outputs = pc_outputs.transpose(0, 1).to(self.options.device)
                pos = pos.transpose(0, 1).to(self.options.device)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                print('Epoch: {}/{}. Batch {}/{}. Loss: {}. Err: {}cm'.format(
                    epoch_idx, n_epochs, batch_idx, len(loader),
                    np.round(loss, 6), np.round(100 * err, 2)))

                # # inspect recurrent weights
                # rec = (self.model.RNN.weight_hh_l0)
                # # l2 norm of recurrent weights
                # rec_norm = torch.norm(rec)
                # print('Recurrent weight norm: {}'.format(rec_norm))
                # dec_norm = torch.norm(self.model.decoder.weight)
                # print('Decoder weight norm: {}'.format(dec_norm))
                # # gradient norm
                # grad_norm = torch.norm(rec.grad.data)
                # print('Gradient norm: {}'.format(grad_norm))

            # Update learning rate
            self.scheduler.step()

            if save and (epoch_idx + 1) % 20 == 0:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                save_ratemaps(self.model, self.trajectory_generator,
                              self.options, step=epoch_idx)

    def train(self, n_epochs: int = 1000, n_steps=10, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(n_epochs):
            for step_idx in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                print(pc_outputs.shape)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                print('Epoch: {}/{}. Step {}/{}. Loss: {}. Err: {}cm'.format(
                    epoch_idx, n_epochs, step_idx, n_steps,
                    np.round(loss, 6), np.round(100 * err, 2)))

                # # inspect recurrent weights
                # rec = (self.model.RNN.weight_hh_l0)
                # # l2 norm of recurrent weights
                # rec_norm = torch.norm(rec)
                # print('Recurrent weight norm: {}'.format(rec_norm))
                # dec_norm = torch.norm(self.model.decoder.weight)
                # print('Decoder weight norm: {}'.format(dec_norm))
                # inc_norm = torch.norm(self.model.encoder.weight)
                # print('Encoder weight norm: {}'.format(inc_norm))
                # # gradient norm
                # grad_norm = torch.norm(rec.grad.data)
                # print('Gradient norm: {}'.format(grad_norm))

            # Update learning rate
            self.scheduler.step()

            if save and (epoch_idx + 1) % 20 == 0:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                save_ratemaps(self.model, self.trajectory_generator,
                              self.options, step=epoch_idx)
