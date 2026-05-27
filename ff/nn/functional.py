import torch
from scipy.optimize import linear_sum_assignment

def calculate_gradient_penalty(real, fake, D):
    """
    Calculate the gradient penalty for the discriminator in WGAN-GP.
    """
    b = real.shape[0]
    n = len(real.shape) - 1
    alpha = torch.rand(b, *[1] * n, device=real.device)
    interpolate = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)
    interpolate_score = D(interpolate)
    gradient = torch.autograd.grad(interpolate_score, interpolate, torch.ones_like(interpolate_score), True, True)[0]
    gradient = gradient.view(b, -1)
    gradient_penalty = torch.mean((torch.norm(gradient, 2, 1) - 1) ** 2)
    return gradient_penalty

def calculate_grad_norm_weight(loss1, loss2, ref_param, eps=1e-4, clamp_range=(0.0, 1000.0)):
    """
    Calculate the weight of the auxiliary loss(loss2) based on the ratio of the gradient norm
    of the two losses with respect to a shared parameter(usually the last layer).
    """
    # Calculate the gradients of losses with respect to the reference parameter
    grad1 = torch.autograd.grad(loss1, ref_param, retain_graph=True)[0]
    grad2 = torch.autograd.grad(loss2, ref_param, retain_graph=True)[0]
    # Calculate the ratio of the gradient norms
    weight = torch.norm(grad1) / (torch.norm(grad2) + eps)
    weight = torch.clamp(weight, clamp_range[0], clamp_range[1]).detach()
    return weight

def get_optimal_transport_noise(x: torch.Tensor) -> torch.Tensor:
    """
    Get optimal transport noise for a given tensor in flow matching training.
    Args:
        x: Tensor to get optimal transport noise for.
    Returns:
        Optimal transport noise.
    """
    b = x.shape[0]
    noise = torch.randn_like(x, dtype=x.dtype, device=x.device)
    with torch.no_grad():
        distances = torch.cdist(x.view(b, -1), noise.view(b, -1)).cpu().detach().numpy()
    _, col_indices = linear_sum_assignment(distances)
    noise = noise[col_indices]
    return noise
