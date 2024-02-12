import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import predictive_coding as pc
import torch.optim as optim
import torch.nn.functional as F

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

    # run the following if this class inherits object.
    # def __call__(self, inp):
    #     return self.forward(inp)

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out

class Sigmoid(nn.Module):
    def forward(self, inp):
        return torch.sigmoid(inp)

    def deriv(self, inp):
        out = self(inp)
        return out * (1 - out)

class Binary(nn.Module):
    def forward(self, inp, threshold=0.):
        return torch.where(inp > threshold, 1., 0.)

    def deriv(self, inp):
        return torch.zeros((1,))

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.RNN_type,
        str(options.Ng),
        options.activation,
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.periodic),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        'data_source', options.data_source,
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID

def ce_loss(output, _target):
    pred = F.softmax(output, dim=-1)
    return -(_target * torch.log(pred)).sum(-1).mean()

def pc_inference(model, inputs, options):
    # print('Predicting...')
    init_tester = pc.PCTrainer(
        model.init_model,
        T=200,
        update_x_at='all',
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={"lr": 1e-2},
        update_p_at='never',
        plot_progress_at=[],
    )

    init_tester.train_on_batch(
        inputs=torch.zeros(options.batch_size, options.Ng).to(options.device),
        loss_fn=ce_loss,
        loss_fn_kwargs={"_target": inputs[1]}, # p0
        is_log_progress=False,
        is_return_results_every_t=False,
    )

    init_state = model.init_model[1].get_x().detach()
    pred = model.predict(inputs[0], init_state)

    return init_state, pred