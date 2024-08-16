import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import ABC
import logging
import distributions
import tensorflow as tf
import logging
import os
from torch.utils.data import TensorDataset

def get_dataset(config, datadir):
    # create distribution
    distribution = distributions.create_distribution(config)
    # load training and evalution samples and save 
    if not tf.io.gfile.exists(datadir):
        logging.info("No saved samples found! Sampling from distribution")
        tf.io.gfile.makedirs(datadir)
        train_samples = distribution.sample(num_samples=config.training.num_train)
        eval_samples = distribution.sample(num_samples=config.eval.num_eval)
        torch.save(train_samples, os.path.join(datadir, 'train_samples.pt'))
        torch.save(eval_samples, os.path.join(datadir, 'eval_samples.pt'))
    else:
        train_samples = torch.load(os.path.join(datadir, 'train_samples.pt'))
        eval_samples = torch.load(os.path.join(datadir, 'eval_samples.pt'))
    # form datasets
    train_ds = TensorDataset(train_samples)
    eval_ds = TensorDataset(eval_samples)
    
    return train_ds, eval_ds, distribution
