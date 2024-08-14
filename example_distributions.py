from SyntheticDistributions.base_distributions import *
import torch


def generate_cov(d, method='principled', components=None):
    # generates a random (invertible) covariance matrix of size d x d
    if method == 'principled':
        if components is None:
            components = torch.rand(d) * 2 + 10 ** (-6)
        A = torch.randn((d, d))
        Q, _ = torch.linalg.qr(A)
        Cov = Q.t() @ torch.diag(components) @ Q
    elif method == 'old':
        while True:
            A = torch.randn((d, d))
            Cov = torch.matmul(A, A.t())
            if torch.linalg.matrix_rank(Cov) == d:
                break
    return Cov


def gaussian_1():
    torch.manual_seed(0)
    return GaussianDistribution(torch.zeros(2), generate_cov(2, components=torch.tensor([2, .5])))


def mixture_distribution_1(weight=.2):
    distribution1 = GaussianDistribution(-5 * torch.ones(2), torch.eye(2))
    distribution2 = GaussianDistribution(5 * torch.ones(2), torch.eye(2))
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution


def mixture_distribution_2(weight=.2):
    distribution1 = GaussianDistribution(-5 * torch.ones(2), .5*torch.eye(2))
    distribution2 = GaussianDistribution(5 * torch.ones(2), 2*torch.eye(2))
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution


def mixture_distribution_3(weight=.2):
    torch.manual_seed(0)
    covariance_1 = generate_cov(2, method='principled', components=torch.tensor([2, .5]))
    covariance_2 = generate_cov(2, method='principled', components=torch.tensor([2, .1]))
    mean_vector = torch.zeros(2)
    # distribution = GaussianDistribution(mean_vector, covariance)
    distribution1 = GaussianDistribution(-5 * torch.ones(2), covariance_1)
    distribution2 = GaussianDistribution(5 * torch.ones(2), covariance_2)
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution


def mixture_distribution_4(weight=.5):
    mean_1 = torch.zeros(2)
    mean_2 = 4*torch.ones(2)
    components_1 = torch.diag(torch.tensor([1., 1.]))
    components_2 = torch.diag(torch.tensor([2., .2]))
    torch.manual_seed(0)
    A = torch.randn((2, 2))
    Q_1, _ = torch.linalg.qr(A)
    B = torch.randn((2, 2))
    Q_2, _ = torch.linalg.qr(B)
    covariance_1 = Q_1 @ components_1 @ Q_1.t()
    covariance_2 = Q_2 @ components_2 @ Q_2.t()
    # covariance_1 = generate_cov(2, method='principled', components=torch.tensor([2, .5]))
    # covariance_2 = generate_cov(2, method='principled', components=torch.tensor([2, .1]))
    distribution1 = GaussianDistribution(mean_1, covariance_1)
    distribution2 = GaussianDistribution(mean_2, covariance_2)
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution

def mixture_distribution_5(weight=.5):
    mean_1 = torch.zeros(2)
    mean_2 = torch.zeros(2)
    components_1 = torch.diag(torch.tensor([.2, 2.]))
    components_2 = torch.diag(torch.tensor([2., .2]))
    torch.manual_seed(0)
    A = torch.randn((2, 2))
    Q_1, _ = torch.linalg.qr(A)
    covariance_1 = Q_1 @ components_1 @ Q_1.t()
    covariance_2 = Q_1 @ components_2 @ Q_1.t()
    # covariance_1 = generate_cov(2, method='principled', components=torch.tensor([2, .5]))
    # covariance_2 = generate_cov(2, method='principled', components=torch.tensor([2, .1]))
    distribution1 = GaussianDistribution(mean_1, covariance_1)
    distribution2 = GaussianDistribution(mean_2, covariance_2)
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution

def mixture_distribution_6():
    weight = .8
    distribution1 = GaussianDistribution(-5 * torch.ones(2), .5*torch.eye(2))
    distribution2 = GaussianDistribution(5 * torch.ones(2), 2*torch.eye(2))
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution

def mixture_distribution_7():
    torch.manual_seed(0)
    weight = .8
    distribution1 = GaussianDistribution(-5 * torch.ones(2), .5*torch.eye(2))
    distribution2 = GaussianDistribution(5 * torch.ones(2), generate_cov(2, method='principled', components=torch.tensor([2, .8])))
    distribution = GaussianMixtureDistribution([distribution1, distribution2], [weight, 1-weight])
    return distribution

def mixture_distribution_8():
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(-5*torch.ones(2), torch.eye(2))
    A = torch.randn((2, 2))
    Q_1, _ = torch.linalg.qr(A)
    components_1 = torch.diag(torch.tensor([.2, 2.]))
    components_2 = torch.diag(torch.tensor([2., .2]))
    covariance_1 = Q_1 @ components_1 @ Q_1.t()
    covariance_2 = Q_1 @ components_2 @ Q_1.t()
    distribution2 = GaussianDistribution(5*torch.ones(2), covariance_1)
    distribution3 = GaussianDistribution(5*torch.ones(2), covariance_2)
    distributions = [distribution1, distribution2, distribution3]
    weights = [.4, .3, .3]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def isotropic_gaussian(dim):
    return GaussianDistribution(torch.zeros(dim), torch.eye(dim))

def anisotropic_gaussian(dim):
    return GaussianDistribution(torch.zeros(dim), torch.diag(torch.linspace(1, 2, dim)))

def high_dim_mixture(dim):
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(-5*torch.ones(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(torch.zeros(dim), torch.diag(torch.linspace(1, 2, dim)))
    distribution3 = GaussianDistribution(5*torch.ones(dim), generate_cov(dim, method='principled', components=torch.linspace(1, 2, dim)))
    distributions = [distribution1, distribution2, distribution3]
    weights = [.4, .3, .3]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def ICASSP_mixture1(dim):
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(torch.zeros(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(5*torch.ones(dim), torch.diag(torch.linspace(.1, 1, dim)))
    distributions = [distribution1, distribution2]
    weights = [.5, .5]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def ICASSP_mixture2(dim):
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(torch.zeros(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(torch.ones(dim), torch.diag(torch.linspace(.5, 1, dim)))
    distributions = [distribution1, distribution2]
    weights = [.5, .5]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def ICASSP_mixture3(dim):
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(torch.zeros(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(5*torch.ones(dim), torch.diag(torch.linspace(.1, 1, dim)))
    distributions = [distribution1, distribution2]
    weights = [.5, .5]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def ICASSP_mixture4(dim):
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(torch.zeros(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(5*torch.ones(dim), torch.diag(torch.linspace(1, 2, dim)))
    distributions = [distribution1, distribution2]
    weights = [.5, .5]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def ICASSP_mixture5(dim):
    torch.manual_seed(0)
    distribution1 = GaussianDistribution(torch.zeros(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(torch.zeros(dim), torch.diag(torch.linspace(1, 2, dim)))
    distributions = [distribution1, distribution2]
    weights = [.5, .5]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution

def laplace_example():
    return LaplaceDistribution(torch.tensor([0.0]), torch.tensor([1.0]))


def gamma_example():
    return GammaDistribution(torch.tensor([10.0]), torch.tensor([1]))

def studentT_example(dof):
    return StudentTDistribution(torch.tensor([dof]))

