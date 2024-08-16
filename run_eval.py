
import os
import logging
import pickle
import tensorflow as tf
import torch
import datasets
import sde_lib
from torch.utils.data import DataLoader
import architectures
import losses
from externals.score_sde_pytorch.models.ema import ExponentialMovingAverage
from externals.score_sde_pytorch.utils import save_checkpoint, restore_checkpoint
import sampling
import distributions
import utils
import matplotlib.pyplot as plt
import plotting

def run_eval(workdir, checkpoint, sde_trial):

    # create directory for results
    eval_dir = os.path.join(workdir, "eval/checkpoint_" + str(checkpoint))
    tf.io.gfile.makedirs(eval_dir)
    trial_dir = os.path.join(eval_dir, sde_trial)
    tf.io.gfile.makedirs(trial_dir)

    # initialize logger
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(trial_dir, ".log")), #TODO: change
        logging.StreamHandler()
    ]
    )

    # load configuration file 
    with open(os.path.join(workdir, 'config.pickle'), 'rb') as handle:
        config = pickle.load(handle)

    # Initialize model.
    model = architectures.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    # Build data iterators
    rand_gen_torch = torch.Generator()
    rand_gen_torch.manual_seed(config.seed)
    datadir = os.path.join(workdir, "data")
    train_ds, eval_ds, distribution = datasets.get_dataset(config, datadir)
    train_dl = DataLoader(train_ds, batch_size=config.training.batch_size, generator=rand_gen_torch, shuffle=True)
    eval_dl = DataLoader(eval_ds, batch_size=config.eval.batch_size, generator=rand_gen_torch, shuffle=True)
    train_iter = iter(train_dl)
    eval_iter = iter(eval_dl) 
    # Create data normalizer and its inverse (needed for Yang Song's code)
    scaler = lambda x: x
    inverse_scaler = lambda x: x

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    # Load model from saved checkpoint
    restore_path = os.path.join(workdir, "checkpoints/checkpoint_" + str(checkpoint) + ".psth")
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    state = restore_checkpoint(restore_path, state, config.device)
    ema.copy_to(model.parameters())
    model.eval()

    # Get sampling function
    sampling_shape = (config.eval.batch_size, config.data.dim)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # Evaluate relative error in score at each noise level on evaluation dataset
    print("Evaluating Error at each timestep ...")
    timesteps = torch.linspace(sampling_eps, 1, 50)
    eval_batch, eval_iter = sample_batch(eval_iter, eval_dl)
    eval_batch = eval_batch.to(config.device).float()
    model_score_fn = utils.get_score_fn(sde, model, train=False, continuous=config.training.continuous)
    rel_errors = torch.zeros(len(timesteps))
    errors = torch.zeros(len(timesteps))
    for idx in range(len(timesteps)):
        # get perturbed data
        timestep = (timesteps[idx] * torch.ones(eval_batch.shape[0], device=config.device)).float() # tensor on device
        z = torch.randn_like(eval_batch)
        mean, std = sde.marginal_prob(eval_batch, timestep)
        perturbed_data = mean + std[:, None] * z
        # compute ground truth scores
        convolved_distribution = distributions.get_convolved_distribution(distribution, std[0].item())
        true_scores = convolved_distribution.score_function(perturbed_data.cpu())
        # get model scores
        model_scores = model_score_fn(eval_batch, timestep)
        # compute relative errors
        rel_errors[idx] = torch.mean(torch.linalg.norm(true_scores - model_scores.cpu(), dim=1) / torch.linalg.norm(true_scores, dim=1)).item()
        errors[idx] = torch.mean(torch.linalg.norm(true_scores - model_scores.cpu(), dim=1)).item()
    torch.save(rel_errors, os.path.join(eval_dir, 'timestep_vs_relerror.pt'))
    plt.plot(timesteps.detach().cpu().numpy(), rel_errors)
    plt.ylim([0, 1])
    plt.ylabel('Relative Error')
    plt.xlabel('Timestep')
    plt.title('Relative Score Matching Error')
    plt.savefig(os.path.join(eval_dir, 'relative_errors.png'))
    plt.close()
    torch.save(errors, os.path.join(eval_dir, 'timestep_vs_error.pt'))
    plt.plot(timesteps.detach().cpu().numpy(), errors)
    plt.ylim([0, 1])
    plt.ylabel('Error')
    plt.xlabel('Timestep')
    plt.title('Score Matching Error')
    plt.savefig(os.path.join(eval_dir, 'errors.png'))
    plt.close()
    print("Finished evaluating Error at each timestep")

    # Visualize samples
    print("Computing and visualizing samples ...")
    sample, n = sampling_fn(model)
    # save samples
    sample = sample.cpu()
    with open(os.path.join(eval_dir, "samples.pt"), "wb") as fout:
        torch.save(sample, fout)
    # compate to ground truth samples
    eval_batch, eval_iter = sample_batch(eval_iter, eval_dl)
    eval_batch = eval_batch.cpu()
    fout = os.path.join(eval_dir, "sample_viz1.png")
    plotting.comp_generative_w_gt(sample, eval_batch, fout)
    fout = os.path.join(eval_dir, "sample_viz2.png")
    plotting.comp_generative_w_gt(sample, eval_batch, fout)
    print("Finished Plotting Samples")

    # Quantitative tests on samples
    print("Conducting quantitative tests on samples ...")
    gen_sample_mean = torch.mean(sample, dim=0)
    gt_sample_mean = torch.mean(eval_batch, dim=0)
    gen_sample_var = torch.var(sample, dim=0)
    gt_sample_var = torch.var(eval_batch, dim=0)
    rel_mean_error = (torch.linalg.norm(gen_sample_mean - gt_sample_mean)/torch.linalg.norm(gt_sample_mean)).item()
    rel_var_error = (torch.linalg.norm(gen_sample_var - gt_sample_var)/torch.linalg.norm(gt_sample_var)).item()
    print("The relative error in population sample mean is {:.4f}, the relative error in population sample variance is {:.4f}".format(rel_mean_error, rel_var_error))
    sample_scores = distribution.score_function(sample)
    gt_scores = distribution.score_function(eval_batch)
    score_mean_error = (torch.linalg.norm(torch.mean(sample_scores**2, dim=0) - torch.mean(gt_scores**2, dim=0))/torch.linalg.norm(torch.mean(gt_scores**2, dim=0))).item()
    print("Relative error in score based test statistic (diagonal of score covariance matrices) was: {:.4f}".format(score_mean_error))
    print("Finished conducting quantitative tests!")



def sample_batch(data_iter, dl):
    try:
        batch = next(data_iter)[0]
    except:
        data_iter = iter(dl)
        batch = next(data_iter)[0]
    return batch, data_iter


if __name__ == "__main__":
    checkpoint = 4
    workdir = 'results/081524ThirdTrain/'
    sde_trial = 'Default'
    run_eval(workdir, checkpoint, sde_trial)