class Experiment:
    def __init__(self, **kwargs):
        self.defaults()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def defaults(self):
        self.MODELDIR = 'resource/logs/'
        self.GAMMA = 0.99
        self.HIDDEN = 128
        self.ENT_DIM = 1014
        self.ENT_DIM_GLOBAL = 3024
        self.LOAD = False
        self.BEST = False
        self.HORIZON = 1000
        self.EPOCHS = 5
        self.EPOCHS_PPO = 10
        self.LAMBDA_ADV_LM = 1.
        self.NPOP = 5
        self.NENT = 5
        self.ENTROPY = .05
        self.ENTROPY_ANNEALING = 0.998
        self.MIN_ENTROPY = .0
        self.DEVICE_OPTIMIZER = 'cpu'
        self.LR = 0.001
        self.LR_DECAY = 0.99998
        self.LR_DECAY_LINEAR = 1e-6
        self.MIN_LR = 0.
        self.LSTM = True
        self.LSTM_PERIOD = 50
        self.LM_MODE = 'sum'  # sum, min
        self.VANILLA_COMA = False
        self.NORM_INPUT = True
        self.INEQ = False
        self.INEQ_ALPHA = 5
        self.INEQ_BETA = .05
        self.INEQ_LAMBDA = 0.95
