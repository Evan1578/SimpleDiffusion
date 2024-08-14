import ml_collections

def default_config():

    config = ml_collections.ConfigDict()
    config.distribution = high_dim_mixture(10)
    config.dim = [10]
    config.dtype = 'synthetic'
    config.noise_levels = noise_level_4()

    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 1000
    training.n_epochs = 1000
    training.num_train = 8000
    training.num_val = 2000
    training.optimizer = 'Adam'
    training.lr = 1e-5
    training.perturb_mag = 0.
    training.use_milestones = False
    training.first_milestone = 10000
    training.milestone_freq = 20
    training.gamma = .99
    training.batch_increases = []
    training.network = 'ToyConditionalModel'
    training.num_hidden1 = 4
    training.num_hidden2 = 2
    training.activation = 'ReLU'
    training.hidden_dim = 512
    training.init_type = 'none'
    training.weight_decay = 0.

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps = 5000
    sampling.tau = 2e-5
    sampling.init_range = [-1, 1]

    return config