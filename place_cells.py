# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy


class PlaceCells(object):

    def __init__(self, options, us=None):
        self.Np = options.Np
        self.sigma = options.place_cell_rf
        self.surround_scale = options.surround_scale
        self.box_width = options.box_width
        self.box_height = options.box_height
        self.is_periodic = options.periodic
        self.DoG = options.DoG
        self.device = options.device
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # Randomly tile place cell centers across environment
        # i.e., Np place cells, each with a randomly chosen center
        # never change the seed, it's important for reproducibility
        np.random.seed(0)
        usx = np.random.uniform(-self.box_width/2, self.box_width/2, (self.Np,))
        usy = np.random.uniform(-self.box_width/2, self.box_width/2, (self.Np,))
        self.centers = torch.tensor(np.vstack([usx, usy]).T)
        # If using a GPU, put on GPU
        self.centers = self.centers.to(self.device)
        # self.centers = torch.tensor(np.load('models/example_pc_centers.npy')).cuda()

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        d = torch.abs(pos[:, :, None, :] - self.centers[None, None, ...]).float()

        if self.is_periodic:
            dx = d[:,:,:,0]
            dy = d[:,:,:,1]
            dx = torch.minimum(dx, self.box_width - dx) 
            dy = torch.minimum(dy, self.box_height - dy)
            d = torch.stack([dx,dy], axis=-1)

        norm2 = (d**2).sum(-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on 
        # average and seems to speed up training.
        outputs = self.softmax(-norm2/(2*self.sigma**2))

        if self.DoG:
            # Again, normalize with prefactor 
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= self.softmax(-norm2/(2*self.surround_scale*self.sigma**2))

            # Shift and scale outputs so that they lie in [0,1].
            min_output,_ = outputs.min(-1,keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)
        return outputs

    def get_nearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''
        _, idxs = torch.topk(activation, k=k)
        pred_pos = self.centers[idxs].mean(-2)
        return pred_pos

    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid
        
        That is, providing an evrironment of size box_width x box_height so that we have a 
        scaffold of points to interpolate onto. This is useful for visualizing place cell
        '''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        pc_outputs = pc_outputs.reshape(-1, self.Np)
        
        T = pc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            # us: place cell centers, shape [Np, 2]
            # pc_outputs: place cell outputs, shape [T, Np]
            # grid: grid points, shape [res**2, 2]
            # grid data basically rewrites the pc_outputs onto the grid
            # but what if pc_outputs are not exatcly on the grid?
            # 
            gridval = scipy.interpolate.griddata(self.centers.cpu(), pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        
        return pc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T

        pos = torch.tensor(pos)

        # Put on GPU if available
        pos = pos.to(self.device)

        #Maybe specify dimensions here again?
        pc_outputs = self.get_activation(pos).reshape(-1,self.Np).cpu()

        C = pc_outputs@pc_outputs.T
        Csquare = C.reshape(res,res,res,res)

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

        return Cmean

class PlaceCells1D(object):

    def __init__(self, options):
        self.Np = options.Np
        self.track_length = options.track_length
        self.sigma = options.place_cell_rf
        self.seed = options.plce_cell_seed
        self.device = options.device
        self.periodic = options.periodic

        # Randomly initialize place cell centers: 1d array of shape [Np].
        np.random.seed(self.seed)
        # self.centers = torch.tensor(np.random.uniform(-self.track_length/2, self.track_length/2, self.Np))
        self.centers = torch.tensor(np.linspace(-self.track_length/2, self.track_length/2, self.Np))
        self.centers = self.centers.to(self.device)

    def get_activation(self, pos):
        '''
        Get the activation of the place cells given the position of the agent

        Inputs:
            pos: (batch_size, seq_len, 1) tensor of the position of the agent

        Outputs:
            activation: (batch_size, seq_len, Np) tensor of the activation of the place cells
        '''

        # Compute distance between position and place cell centers: [batch_size, sequence_length, Np].
        d = torch.abs(pos[:, :, None, :] - self.centers[None, None, :, None]).float().squeeze(-1)
        if self.periodic:
            d = torch.min(d, self.track_length - d)

        if self.sigma is not None:
            # use gaussian activation
            activation = torch.exp(-d**2 / (2 * self.sigma**2))
        else:
            # use one hot encoding
            min_vals, _ = torch.min(d, dim=-1, keepdim=True)
            activation = (d == min_vals).float()

        return activation

    def get_nearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 1].
        '''
        # top-k place cells along the last dimension; idx: [batch_size, sequence_length, k]
        _, idxs = torch.topk(activation, k=k)
        # centers[idxs] will slice the centers along the last dimension using the indices in idxs: [batch_size, sequence_length, k]
        pred_pos = self.centers[idxs].mean(-1, keepdim=True)
        return pred_pos