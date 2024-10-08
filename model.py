# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils as utils
import predictive_coding as pc


class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.loss = options.loss
        self.truncating = options.truncating

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(
            input_size=2,
            hidden_size=self.Ng,
            nonlinearity=options.rec_activation,
            bias=False,
            batch_first=True,
        )
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=True)
        
        if options.out_activation == 'softmax':
            self.out_activation = torch.nn.Softmax(dim=-1)
        elif options.out_activation == 'tanh':
            self.out_activation = torch.nn.Tanh()

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        if self.truncating == 0:
            v, p0 = inputs
            init_state = self.encoder(p0)[None]
            g,_ = self.RNN(v, init_state)
            return g

        else:
            total_g = []
            vs, p0 = inputs
            seq_len = vs.size(1)
            h = self.encoder(p0)[None]
            for k in range(seq_len):
                v = vs[:, k:k+1] # bsz, 1, 2
                g, h = self.RNN(v, h) # g: bsz, 1, Ng
                h = h.detach()
                total_g.append(g)    
            total_g = torch.cat(total_g, dim=1)            
            return total_g # bsz, seq_len, Ng

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
        y = pc_outputs
        preds = self.out_activation(self.predict(inputs))
        if self.loss == 'CE':
            loss = -(y * torch.log(preds + 1e-9)).sum(-1).mean()
        elif self.loss == 'MSE':
            loss = torch.sum((preds - y) ** 2, -1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        preds = F.softmax(preds, dim=-1)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

class HierarchicalPCN(nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)
        self.mu = nn.Parameter(torch.zeros((options.Ng)))

        # sparse penalty
        self.sparse_z = options.lambda_z_init
        if options.out_activation == 'softmax':
            self.out_activation = utils.Softmax()
        elif options.out_activation == 'tanh':
            self.out_activation = utils.Tanh()
        elif options.out_activation == 'sigmoid':
            self.out_activation = utils.Sigmoid()
        self.loss = options.loss

    def set_sparsity(self, sparsity):
        self.sparse_z = sparsity

    def set_nodes(self, inp):
        # intialize the value nodes
        self.z = self.mu.clone()
        self.x = inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def update_err_nodes(self):
        self.err_z = self.z - self.mu
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-8)
        else:
            self.err_x = self.x / (pred_x + 1e-8) + (1 - self.x) / (1 - pred_x + 1e-8)

    def inference_step(self, inf_lr):
        Wout = self.Wout.weight.clone().detach()
        if isinstance(self.out_activation, utils.Softmax):
            delta = self.err_z - (self.out_activation.deriv(self.Wout(self.z)) @ self.err_x.unsqueeze(-1)).squeeze(-1) @ Wout
        else:
            delta = self.err_z - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
        delta += self.sparse_z * torch.sign(self.z) 
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, inp):
        self.set_nodes(inp)
        for itr in range(inf_iters):
            with torch.no_grad():
                self.inference_step(inf_lr)
            self.update_err_nodes()

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        if self.loss == 'CE':
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == 'BCE':
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        return energy, obs_loss

class TemporalPCN(nn.Module):
    """Multi-layer tPC class, using autograd"""
    def __init__(self, options):
        super(TemporalPCN, self).__init__()
        self.Wr = nn.Linear(options.Ng, options.Ng, bias=False)
        self.Win = nn.Linear(options.Nv, options.Ng, bias=False)
        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)

        if options.no_velocity:
            self.Win.weight.data.fill_(0)
            self.Win.weight.requires_grad = False

        self.sparse_z = options.lambda_z
        self.weight_decay = options.weight_decay
        if options.out_activation == 'softmax':
            self.out_activation = utils.Softmax()
        elif options.out_activation == 'tanh':
            self.out_activation = utils.Tanh()
        elif options.out_activation == 'sigmoid':
            self.out_activation = utils.Sigmoid()
        
        if options.rec_activation == 'tanh':
            self.rec_activation = utils.Tanh()
        elif options.rec_activation == 'relu':
            self.rec_activation = utils.ReLU()
        elif options.rec_activation == 'sigmoid':
            self.rec_activation = utils.Sigmoid()

        self.loss = options.loss

    def set_nodes(self, v, prev_z, p):
        """Set the initial value of the nodes;

        In particular, we initialize the hiddden state with a forward pass.
        
        Args:
            v: velocity input at a particular timestep in stimulus
            prev_z: previous hidden state
            p: place cell activity at a particular timestep in stimulus
        """
        self.z = self.g(v, prev_z)
        self.x = p.clone()
        self.update_err_nodes(v, prev_z)

    def update_err_nodes(self, v, prev_z):
        self.err_z = self.z - self.g(v, prev_z)
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-9)
        else:
            self.err_x = self.x / (pred_x + 1e-9) - (1 - self.x) / (1 - pred_x + 1e-9)

    def g(self, v, prev_z):
        return self.rec_activation(self.Wr(prev_z) + self.Win(v))
        
    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def inference_step(self, inf_lr, v, prev_z):
        """Take a single inference step"""
        Wout = self.Wout.weight.detach().clone() # shape [Np, Ng]
        if isinstance(self.out_activation, utils.Softmax):
            delta = self.err_z - (self.out_activation.deriv(self.Wout(self.z)) @ self.err_x.unsqueeze(-1)).squeeze(-1) @ Wout
        else:
            delta = self.err_z - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, v, prev_z, p):
        """Run inference on the hidden state"""
        self.set_nodes(v, prev_z, p)
        for i in range(inf_iters):
            with torch.no_grad(): # ensures self.z won't have grad when we call backward
                self.inference_step(inf_lr, v, prev_z)
            self.update_err_nodes(v, prev_z)
                
    def get_energy(self):
        """Returns the average (across batches) energy of the model"""
        if self.loss == 'CE':
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == 'BCE':
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        energy += self.weight_decay * (torch.mean(self.Wr.weight**2))

        return energy, obs_loss

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
        self.Win = nn.Linear(options.Nv, options.Ng, bias=False)
        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)

        self.nonlin = options.activation

        self.sparse_z = options.lambda_z
        self.hidden_size = options.Ng

        # for initializing the hidden state
        self.mu = nn.Parameter(torch.zeros(options.Ng))
        # indicates whether we are performing inference at the first step
        self.init = True

        self.weight_decay = options.weight_decay

        self.softmax = utils.Softmax()
        self.sigmoid = utils.Sigmoid()
        self.ce_loss = options.ce_loss
    
    def forward(self, v):
        """v: velocity input at a particular timestep in stimulus"""
        pred_z = self.g(v)
        pred_x = self.predict(pred_z)
        return pred_z, pred_x

    def g(self, v):
        if self.init:
            # broadcasting the mean to the batch size; v will be zero at the first step
            pred_z = self.mu + self.nonlin(self.Win(v))
        else:
            pred_z = self.nonlin(self.Wr(self.prev_z) + self.Win(v))
            # pred_z = self.Wr(self.nonlin(self.prev_z)) + self.Win(self.nonlin(v))

        self.set_init(False)
        return pred_z

    def predict(self, z):
        return self.softmax(self.Wout(z))
        # return self.sigmoid(self.Wout(z))
        # return self.Wout(self.nonlin(z))

    def set_init(self, init):
        self.init = init

    def set_prev_z(self, prev_z):
        self.prev_z = prev_z

    def init_hidden(self, bsz):
        """Initializing prev_z randomly"""
        self.set_init(True)
        self.prev_z = nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size)).to(self.Wr.weight.device)

    def update_errs(self, x, v):
        """Update the prediction errors
        
        Inputs:
            x: place cell activity at a particular timestep in stimulus
            v: velocity input at a particular timestep in stimulus
        """
        self.err_z = self.z - self.g(v)
        self.err_x = x / (self.predict(self.z) + 1e-8) if self.ce_loss else x - self.predict(self.z)
    
    def update_nodes(self, inf_lr, x, v):
        """Update the values nodes and error nodes"""
        self.update_errs(x, v)
        Wout = self.Wout.weight.detach().clone() # shape [Np, Ng]
        delta_z = self.err_z - (self.softmax.deriv(self.z @ Wout.t()) @ self.err_x.unsqueeze(-1)).squeeze(-1) @ Wout
        # delta_z = self.err_z - (self.sigmoid.deriv(self.z @ Wout.t()) * self.err_x) @ Wout
        # delta_z = self.err_z - self.nonlin.deriv(self.z) * (self.err_x @ Wout)
        delta_z += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta_z

    def inference(self, inf_iters, inf_lr, x, v):
        """Run inference on the hidden state"""
        with torch.no_grad():
            # initialize the current hidden state with a forward pass from the randomly initialized prev_z
            self.z = self.g(v)
            for i in range(inf_iters):
                self.update_nodes(inf_lr, x, v)

            # update the previous hidden state with the current hidden state
            # note that this should be kept constant throughout inference
            self.prev_z = self.z.clone() 
                
    def get_energy(self, x, v):
        """returns the average (across batches) energy of the model
        
        Inputs:
            x: place cell activity at a particular timestep in stimulus
            v: velocity input at a particular timestep in stimulus
        """
        err_z = self.z - self.g(v)
        self.hidden_loss = torch.sum(err_z**2, dim=-1).mean()
        if self.ce_loss:
            self.obs_loss = -torch.sum(torch.log(self.predict(self.z) + 1e-8) * x, dim=-1).mean()
        else:
            self.obs_loss = torch.sum((x - self.predict(self.z))**2, dim=-1).mean()
        energy = self.hidden_loss + self.obs_loss
        energy += self.weight_decay * torch.sum(self.Wr.weight**2)

        return energy

class MultilayerPCN(nn.Module):
    def __init__(self, nodes, nonlin, lamb=0., use_bias=False, relu_inf=True):
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
        self.relu_inf = relu_inf

        if nonlin == 'tanh':
            nonlin = utils.Tanh()
        elif nonlin == 'teLU':
            nonlin = utils.ReLU()
        elif nonlin == 'linear':
            nonlin = utils.Linear()
        self.nonlins = [nonlin] * (self.n_layers - 1)
        self.use_bias = use_bias

        # initialize nodes
        self.val_nodes = [[] for _ in range(self.n_layers)]
        # self.preds = [[] for _ in range(self.n_layers)]
        self.errs = [[] for _ in range(self.n_layers)]

        # sparse penalty
        self.lamb = lamb

    def set_sparsity(self, sparsity):
        self.lamb = sparsity

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
            self.val_nodes[l] = F.relu(self.val_nodes[l] + inf_lr * delta) if self.relu_inf else self.val_nodes[l] + inf_lr * delta

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