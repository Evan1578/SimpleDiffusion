import matplotlib.pyplot as plt
import torch

def comp_generative_w_gt(generative_samples, ground_truth_samples, fout):
    """
    generative_samples (torch.Tensor): tensor of size batch_size x D containing generative modeling samples
    ground_truth_samples (torch.Tensor): tensor of size batch_size x D containing ground truth samples
    fout (str): file to save plot to 
    """
    assert generative_samples.shape == ground_truth_samples.shape # dimensions should be the same
    # generate random projection matrix
    proj_matrix = torch.randn(generative_samples.shape[1], 2)
    # take projection
    proj_gen_samples  = torch.matmul(generative_samples, proj_matrix)
    proj_gt_samples = torch.matmul(ground_truth_samples, proj_matrix)
    # plot results
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title("Random 2D Projections of Samples")
    ax.scatter(torch.minimum(torch.maximum(proj_gen_samples[:, 0], torch.tensor([-50])), torch.tensor([50])), 
               torch.minimum(torch.maximum(proj_gen_samples[:, 1], torch.tensor([-50])), torch.tensor([50])), label='Generative Samples')
    ax.scatter(torch.minimum(torch.maximum(proj_gt_samples[:, 0], torch.tensor([-50])), torch.tensor([50])), 
               torch.minimum(torch.maximum(proj_gt_samples[:, 1], torch.tensor([-50])), torch.tensor([50])), label = 'Ground Truth Samples')
    ax.legend()
    ax.grid(True)
    plt.savefig(fout)
    plt.close()