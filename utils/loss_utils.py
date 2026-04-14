#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
from kornia.filters import bilateral_blur
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# im: 3,H,W, with grad
def bilateral_smooth_img_loss(im: torch.Tensor, refl_map: torch.Tensor, cluster_map: torch.Tensor = None):
    REFL_THRESH = 0.05
    
    # BIG BUG FIX: The original code averaged the X,Y,Z Normal vector and checked if it was > 0.05!
    # This mathematically excluded the ENTIRE bottom hemisphere of all geometry from getting smoothed!
    # We now properly use the actual reflection map.
    msk = refl_map[0] > REFL_THRESH
    if not torch.any(msk): return 0
    
    cim = im.detach().clone() # 3, H, W
    
    # CLUSTER ISOLATION HACK:
    # Kornia's bilateral_blur blends pixels if their color difference is within 75/255.
    # By shifting each cluster's normal vector mathematical value up by a massive offset (100 * ID),
    # the Bilateral filter sees clusters as infinitely different colors, and will violently 
    # refuse to blur normal boundaries across different clusters!
    if cluster_map is not None:
        cim += (cluster_map * 100.0)

    cim[:, ~msk] = -999999.0 
    
    # Blur the full 3xHxW geometry tensor
    smoothed_im = bilateral_blur(cim[None], (11,11), 75/255, (10,10))[0]
    
    # Strip the offset out before comparing losses
    if cluster_map is not None:
        smoothed_im -= (cluster_map * 100.0)
        
    loss = l2_loss(im[:, msk], smoothed_im[:, msk])
    return loss

# im: 3,H,W, with grad
gBlur = None
def smooth_img_loss(im: torch.Tensor):
    global gBlur
    if gBlur is None:
        gBlur = GaussianBlur(9, 4.0).cuda()
    im_smooth = gBlur(im[None].detach())[0]
    loss = l2_loss(im, im_smooth)
    return loss