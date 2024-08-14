from abc import ABC

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.laplace import Laplace
from torch.distributions.gamma import Gamma
from torch.distributions.studentT import StudentT
import numpy as np


class GaussianDistribution(MultivariateNormal, ABC):

    # Pytorch multivariate normal distribution with added score function
    def __init__(self, mean, covariance_matrix):
        super().__init__(mean, covariance_matrix=covariance_matrix)

    def score_function(self, x):
        # input: Pytorch tensor of size batch_size x d
        # output: Pytorch tensor of size batch_size x d
        return -torch.matmul(x - self.mean[None, :], self.precision_matrix)

    def sample(self, num_samples=1):
        sample_shape = torch.Size([num_samples])
        return super().sample(sample_shape=sample_shape)

class LaplaceDistribution(Laplace, ABC):

    # Pytorch Laplace distribution with added score function
    def __init__(self, mean, scale):
        super().__init__(mean, scale)

    def score_function(self, x):
        return (-1/self.scale)*(x > self.loc) + (1/self.scale)*(x <= self.loc)

    def sample(self, num_samples=1):
        sample_shape = torch.Size([num_samples])
        return super().sample(sample_shape=sample_shape)

class GammaDistribution(Gamma, ABC):

    # Pytorch Gamma distribution with added score function
    def __init__(self, concentration, rate):
        super().__init__(concentration, rate)

    def score_function(self, x):
        if self.concentration == 1:
            return -self.rate*torch.ones_like(x)
        else:
            return (self.concentration - 1) / torch.clamp(x, min=10**(-5)) - self.rate

    def sample(self, num_samples=1):
        sample_shape = torch.Size([num_samples])
        return super().sample(sample_shape=sample_shape)

class StudentTDistribution(StudentT, ABC):

    # Pytorch student T distribution with added score function
    def __init__(self, df):
        super().__init__(df)

    def score_function(self, x):
        return ((-1.0 - self.df)/self.df)*x/(1 + torch.pow(x, 2)/self.df)

    def sample(self, num_samples=1):
        sample_shape = torch.Size([num_samples])
        return super().sample(sample_shape=sample_shape)


class GaussianMixtureDistribution:
    # create a distribution as weighted sum of mixture of two distributions
    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.num_distributions = len(distributions)
        self.weights = weights

    def log_prob(self, x):
        exp_of_term = 0
        for idx, distribution in enumerate(self.distributions):
            exp_of_term += self.weights[idx] * torch.exp(distribution.log_prob(x))
        return torch.log(exp_of_term)

    def score_function(self, x):
        log_probs = torch.zeros(len(self.distributions), x.shape[0])
        for idx, distribution in enumerate(self.distributions):
            log_probs[idx, :] = distribution.log_prob(x)
        normalization_term = torch.max(log_probs, 0)[0]
        constant_term = 0
        for idx, distribution in enumerate(self.distributions):
            constant_term += self.weights[idx] * torch.exp(distribution.log_prob(x) - normalization_term)
        val = torch.zeros_like(x)
        for idx, distribution in enumerate(self.distributions):
            prob_correction = torch.unsqueeze(torch.exp(distribution.log_prob(x) - normalization_term)/constant_term, 1)
            val += self.weights[idx] * prob_correction * distribution.score_function(x)
        return val

    def score_function_2(self, x):
        term1 = 1 / (torch.max(torch.exp(self.log_prob(x)), torch.tensor([10 ** (-25)])))
        other_terms = 0
        for idx, distribution in enumerate(self.distributions):
            other_terms += self.weights[idx] * torch.unsqueeze(torch.exp(distribution.log_prob(x)),
                                                               dim=1) * distribution.score_function(x)
        return torch.unsqueeze(term1, dim=1) * other_terms

    def sample(self, num_samples=1):
        which_dist = torch.tensor(
            np.random.choice(self.num_distributions, size=(num_samples, 1), replace=True, p=self.weights))
        samples = []
        for idx, distribution in enumerate(self.distributions):
            samples.append((which_dist == idx) * distribution.sample(num_samples=num_samples))
        cross = torch.stack(samples)
        return cross.sum(dim=0)
