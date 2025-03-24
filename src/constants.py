import src.utils as utils

PERIODIC = False
DEVICE = "cuda"
ONED = False
WEIGHT_DECAY = 1e-4
DECAY_STEP_SIZE = 10
DECAY_RATE = 0.9
LAMBDA_Z = 0
LAMBDA_Z_INIT = 0
SURROUND_SCALE = 2

ACTIVATION_FUNCS = {
    "softmax": utils.Softmax(),
    "tanh": utils.Tanh(),
    "sigmoid": utils.Sigmoid(),
    "relu": utils.ReLU(),
}