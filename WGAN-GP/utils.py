import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):
    """
        Compute Gradient Penalty for WGAN-GP
    """

    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1,C, H,W).to(device)
    
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    #calculate critic score
    mixed_score = critic(interpolated_images)

    #compute gradient of mixed_score w.r.t interpolated images
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_score,
        grad_outputs = torch.ones_like(mixed_score),
        create_graph = True,
        retain_graph = True,
    )[0]

    
    gradient = gradient.view(gradient.shape[0], -1)
    #compute L2 norm of gradient across dim=1
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    


