"""
Script based on code originally ported from https://github.com/yang-song/score_sde_pytorch/blob/main/run_lib.py
"""
import training_configs
import tensorflow as tf
import os
from torch.utils import tensorboard
import torch
import logging
import numpy as np
import utils
import architectures
from ema import ExponentialMovingAverage
import sde_lib
import losses
from utils import save_checkpoint, restore_checkpoint
import sampling
from torch.utils.data import DataLoader
import datasets
import plotting
import pickle

def run_train(config, workdir):

    # create workdir
    tf.io.gfile.makedirs(workdir)

    # setup logging
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(workdir, ".log")),
        logging.StreamHandler()
    ]
    )
    
    # fix seeds
    torch.manual_seed(config.seed)

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)


    with open(os.path.join(workdir, 'config.pickle'), "wb") as handle:
        pickle.dump(config, handle)

    # Initialize model.
    score_model = architectures.create_model(config) # TODO: must include is_energy param
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

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


    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting
    is_energy = config.model.is_energy
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=True, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting, is_energy=is_energy)
    eval_step_fn1 = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                        reduce_mean=True, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting, is_energy=is_energy)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.eval.batch_size, config.data.dim)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, is_energy=is_energy) 

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Get batch
        batch, train_iter = sample_batch(train_iter, train_dl)
        batch = scaler(batch).to(config.device).float()
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, Training Loss (DSM): %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss_dsm", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch, eval_iter = sample_batch(eval_iter, eval_dl)
            eval_batch = scaler(eval_batch).to(config.device).float()
            eval_loss = eval_step_fn1(state, eval_batch)
            logging.info("step: %d, Eval Loss (DSM): %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss_dsm", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        # step != 0 and 
        if (step % config.training.snapshot_freq == 0 or step == num_train_steps):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.psth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                # sample batch
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                # save samples
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                tf.io.gfile.makedirs(this_sample_dir)
                sample = sample.cpu()
                with open(os.path.join(this_sample_dir, "samples.pt"), "wb") as fout:
                    torch.save(sample, fout)
                # compate to ground truth samples
                fout = os.path.join(this_sample_dir, "samples.png")
                eval_batch, eval_iter = sample_batch(eval_iter, eval_dl)
                plotting.comp_generative_w_gt(sample, eval_batch.cpu(), fout)

def sample_batch(data_iter, dl):
    try:
        batch = next(data_iter)[0]
    except:
        data_iter = iter(dl)
        batch = next(data_iter)[0]
    return batch, data_iter


if __name__ == "__main__":
    config = training_configs.default_config()
    workdir = 'results/082124Energy2'
    run_train(config, workdir)