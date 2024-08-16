import ml_collections
import torch

def default_config():

    config = ml_collections.ConfigDict()

    config.data = data = ml_collections.ConfigDict()
    data.distribution_name = 'gaussian_mixture'
    data.dim = 10

    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 8000
    training.n_iters = 20000
    training.num_train = 15000
    training.num_val = 5000
    training.log_freq = 50
    training.eval_freq = 100
    training.snapshot_freq = 5000
    training.sde = 'vesde'
    training.continuous = True
    training.snapshot_freq_for_preemption = 5000
    training.likelihood_weighting = False
    training.snapshot_sampling = True

    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 8000
    eval.num_eval = 20000


    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 1e-5
    optim.beta1 = .9
    optim.eps = 1e-8
    optim.amsgrad = False
    optim.warmup = 0.
    optim.weight_decay = 0.
    optim.grad_clip = -1.

    config.model = model = ml_collections.ConfigDict()
    model.name = 'ToyConditionalModel'
    model.sigma_min = .01
    model.sigma_max = 10
    model.num_scales = 1000
    model.ema_rate = 0.9999
    model.num_hidden1 = 4
    model.num_hidden2 = 2
    model.activation = 'ReLU'
    model.hidden_dim = 512
    model.init_type = 'none'
    model.weight_decay = 0.

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'
    sampling.probability_flow = False
    sampling.snr = 0.15
    sampling.n_steps_each = 1
    sampling.noise_removal = True

    config.seed = 0
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(config.device)

    return config