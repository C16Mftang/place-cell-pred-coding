# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils as utils
import predictive_coding as pc

# a utility model for unsupervised PC training
class Bias(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        # x must be a zero tensor
        return self.bias + x

class tPCLayer(nn.Module):
    def __init__(self, size_in, size_hidden):
        super(tPCLayer, self).__init__()
        # Initialize the Win and Wr linear layers according to the specified shapes
        if size_in > 0:
            self.Win = nn.Linear(size_in, size_hidden, bias=False)
        self.Wr = nn.Linear(size_hidden, size_hidden, bias=False)
        # determine whether there is an external input
        self.is_in = True if size_in > 0 else False

    def forward(self, inputs):
        # input: a tuple of two tensors (hidden (from the previous time step), velocity input)
        # Compute Win(input) + Wr(hidden) and return the result
        return self.Win(inputs[1]) + self.Wr(inputs[0]) if self.is_in else self.Wr(inputs[0])

class tPC(nn.Module):
    def __init__(self, options):
        super(tPC, self).__init__()
        self.size_in = options.size_in
        self.Ng = options.Ng
        self.Np = options.Np
        if options.activation == 'tanh':
            self.activation = nn.Tanh()
        elif options.activation == 'relu':
            self.activation = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        # the decoder from hidden state to place cell activations
        self.decoder = nn.Sequential(
            nn.Linear(self.Ng, self.Np, bias=False),
            pc.PCLayer(),
            self.softmax,
        )

        # the unsupervised model at initialization of sequences
        # self.init_model = nn.Sequential(
        #     Bias(self.Ng),
        #     pc.PCLayer(),
        #     self.decoder,
        # )

        # the recurrent layer
        self.rec_layer = nn.Sequential(
            tPCLayer(self.size_in, self.Ng),
            pc.PCLayer(),
            self.activation,
        )

        self.tpc = nn.Sequential(
            self.rec_layer,
            self.decoder,
        )

        self.options = options

    def forward(self, inputs):
        '''inputs: a tuple of two tensors (hidden (from the previous time step), velocity input)'''
        return self.tpc(inputs)

    def g(self, inputs, init_state):
        '''
        Compute grid cell activations.
        To get initial state, we first need to run inference on decoder,
        and extract the hidden state from the decoder.
        Here we assume this has already been done externally.

        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
            init_state: Initial hidden state, with shape [batch_size, Ng], 
                which has been inferred from the decoder externally.
        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        gcs = torch.zeros((self.options.sequence_length, self.options.batch_size, self.options.Ng))
        g = init_state
        for k in range(self.options.sequence_length):
            v = inputs[k]
            g = self.tpc[0]((g, v)) # infer by forward pass through tPCLayer
            gcs[k] = self.activation(g)
        return gcs.to(self.options.device)

    def predict(self, inputs, init_state):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
            init_state: Initial hidden state, with shape [batch_size, Ng], 
                which has been inferred from the decoder externally.

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = torch.zeros((self.options.sequence_length, self.options.batch_size, self.options.Np))
        gcs = self.g(inputs, init_state)
        for k in range(self.options.sequence_length):
            place_preds[k] = self.decoder(gcs[k])
        return place_preds

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=options.activation,
                                bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=True)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss = torch.nn.MSELoss()

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        # y = torch.nn.functional.one_hot(
        #     torch.argmax(pc_outputs, -1),
        #     num_classes=self.Np,
        # ).float()
        y = pc_outputs
        preds = self.softmax(self.predict(inputs))
        # print(preds[0])
        # yhat = self.softmax(self.predict(inputs))
        loss = -(y*torch.log(preds)).sum(-1).mean()
        # loss = self.loss(self.softmax(preds), y)
        # loss = (preds - y).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

class HierarchicalPCN(nn.Module):
    def __init__(self, nodes, nonlin, lamb=0., use_bias=False):
        super().__init__()
        self.n_layers = len(nodes)
        self.layers = nn.Sequential()
        for l in range(self.n_layers-1):
            self.layers.add_module(f'layer_{l}', nn.Linear(
                in_features=nodes[l],
                out_features=nodes[l+1],
                bias=use_bias,
            ))

        self.mem_dim = nodes[0]
        self.memory = nn.Parameter(torch.zeros((nodes[0],)))

        if nonlin == 'Tanh':
            nonlin = utils.Tanh()
        elif nonlin == 'ReLU':
            nonlin = utils.ReLU()
        elif nonlin == 'Linear':
            nonlin = utils.Linear()
        self.nonlins = [nonlin] * (self.n_layers - 1)
        self.use_bias = use_bias

        # initialize nodes
        self.val_nodes = [[] for _ in range(self.n_layers)]
        # self.preds = [[] for _ in range(self.n_layers)]
        self.errs = [[] for _ in range(self.n_layers)]

        # sparse penalty
        self.lamb = lamb

    def get_inf_losses(self):
        return self.inf_losses # inf_iters,

    def update_err_nodes(self):
        for l in range(0, self.n_layers):
            if l == 0:
                self.errs[l] = self.val_nodes[l] - self.memory
            else:
                preds = self.layers[l-1](self.nonlins[l-1](self.val_nodes[l-1]))
                self.errs[l] = self.val_nodes[l] - preds

    def update_val_nodes(self, inf_lr):
        for l in range(0, self.n_layers-1):
            derivative = self.nonlins[l].deriv(self.val_nodes[l])
            # sparse penalty
            penalty = self.lamb if l == 0 else 0.
            delta = -self.errs[l] - penalty * torch.sign(self.val_nodes[l]) + derivative * torch.matmul(self.errs[l+1], self.layers[l].weight)
            self.val_nodes[l] = F.relu(self.val_nodes[l] + inf_lr * delta)

    def set_nodes(self, batch_inp):
        # computing val nodes
        self.val_nodes[0] = self.memory.clone()
        for l in range(1, self.n_layers-1):
            self.val_nodes[l] = self.layers[l-1](self.nonlins[l-1](self.val_nodes[l-1]))
        self.val_nodes[-1] = batch_inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def inference(self, batch_inp, n_iters, inf_lr):
        self.set_nodes(batch_inp)
        self.inf_losses = []
        
        for itr in range(n_iters):
            with torch.no_grad():
                self.update_val_nodes(inf_lr)
            self.update_err_nodes()
            self.inf_losses.append(self.get_energy().item())

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        total_energy = 0
        for l in range(self.n_layers):
            total_energy += torch.sum(self.errs[l] ** 2) # average over batch and feature dimensions
        return total_energy


class PredSparseCoding(nn.Module):
    """
    Predictive coding implementation of the sparse coding model
    """
    def __init__(self, hidden_size, output_size, nonlin='linear', lambda_z=0.):
        super(PredSparseCoding, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)
        if nonlin == 'Tanh':
            self.nonlin = utils.Tanh()
        elif nonlin == 'ReLU':
            self.nonlin = utils.ReLU()
        elif nonlin == 'Linear':
            self.nonlin = utils.Linear()
        else:
            raise ValueError("no such nonlinearity!")

        self.lambda_z = lambda_z
        
    def forward(self):
        return self.Wout(self.nonlin(self.z))
    
    def get_hidden(self):
        return self.z.clone()
    
    def get_inf_losses(self):
        return self.inf_losses # inf_iters,
    
    def set_z(self, bsz):
        """Initializing the hidden state with batch size bsz"""
        self.z = nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size)).to(self.Wout.weight.device)
    
    def weight_norm(self):
        # in-place normalization of weight parameters
        with torch.no_grad():
            self.Wout.weight.div_(torch.norm(self.Wout.weight, dim=0, keepdim=True)) 

    def update_errs(self, x):
        """Update the prediction errors"""
        err = x - self.forward()
        return err

    def update_z(self, x, inf_lr):
        """Update the hidden state
        sparse_z: sparsity constraint on the hidden state
        """
        err = self.update_errs(x) # bsz x output_size
        delta_z = -self.nonlin.deriv(self.z) * torch.matmul(err, self.Wout.weight.detach().clone())
        delta_z += self.lambda_z * torch.sign(self.z)
        self.z -= inf_lr * delta_z
        
    def inference(self, x, inf_iters, inf_lr):
        """Run inference on the hidden state"""
        self.set_z(x.shape[0])
        self.inf_losses = []
        with torch.no_grad():
            for _ in range(inf_iters):
                self.update_z(x, inf_lr)
                # logging the energy during inference
                self.inf_losses.append(self.get_energy(x).item())
                
    def get_energy(self, x):
        """returns the average (across batches) energy of the model"""
        err = self.update_errs(x)
        # use mean to ensure that energy is independent of batch size and output size
        energy = torch.sum(err ** 2) + self.lambda_z * torch.sum(torch.abs(self.z))
        return energy

class MultilayertPC(nn.Module):
    """Multi-layer tPC class, using autograd"""
    def __init__(self, options):
        super(MultilayertPC, self).__init__()
        self.Wr = nn.Linear(options.Ng, options.Ng, bias=False)
        self.Win = nn.Linear(2, options.Ng, bias=False)
        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)
        self.first_encoder = nn.Linear(options.Np, options.Ng, bias=False)

        if options.activation == 'linear':
            self.nonlin = utils.Linear()
        elif options.activation == 'tanh':
            self.nonlin = utils.Tanh()
        else:
            raise ValueError("no such nonlinearity!")

        self.sparse_z = options.sparse_z
        self.weight_decay = options.weight_decay
    
    def forward(self, v):
        """v: velocity input at a particular timestep in stimulus"""
        pred_z = self.Wr(self.nonlin(self.prev_z)) + self.Win(self.nonlin(v))
        pred_x = self.Wout(self.nonlin(pred_z))
        return pred_z, pred_x

    def init_hidden(self, p0):
        """Initializing prev_z with a linear projection of the first place cell activity p0"""
        # self.prev_z = nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size)).to(self.Wr.weight.device)
        self.prev_z = self.first_encoder(p0)

    def update_errs(self, x, v):
        """Update the prediction errors
        
        Inputs:
            x: place cell activity at a particular timestep in stimulus
            v: velocity input at a particular timestep in stimulus
        """
        pred_z, _ = self.forward(v)
        pred_x = self.Wout(self.nonlin(self.z))
        err_z = self.z - pred_z
        err_x = x - pred_x
        return err_z, err_x
    
    def update_nodes(self, inf_lr, x, v, update_x=False):
        """Update the values nodes and error nodes"""
        err_z, err_x = self.update_errs(x, v)
        delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
        delta_z += self.sparse_z * torch.sign(self.z)
        self.z -= inf_lr * delta_z
        if update_x:
            delta_x = err_x
            x -= inf_lr * delta_x

        # update the previous hidden state with the current hidden state
        self.prev_z = self.z.clone()  

    def inference(self, inf_iters, inf_lr, x, v, update_x=False):
        """run inference on the hidden state"""
        with torch.no_grad():
            # initialize the current hidden state with a forward pass from the randomly initialized prev_z
            self.z, _ = self.forward(v)
            for i in range(inf_iters):
                self.update_nodes(inf_lr, x, v, update_x)

    def set_latents(self, z):
        """setting the hidden state to a particular value"""
        self.z = z.clone().detach()

    def get_latents(self):
        return self.z.clone().detach()
                
    def get_energy(self, x, v):
        """returns the average (across batches) energy of the model
        
        Inputs:
            x: place cell activity at a particular timestep in stimulus
            v: velocity input at a particular timestep in stimulus
        """
        err_z, err_x = self.update_errs(x, v)
        self.hidden_loss = torch.sum(err_z**2, dim=-1).mean()
        self.obs_loss = torch.sum(err_x**2, dim=-1).mean()
        energy = self.hidden_loss + self.obs_loss
        energy += self.weight_decay * torch.sum(self.Wr.weight**2)
        return energy