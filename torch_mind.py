
#!/usr/bin/env python3
"""
mind3d.py
==========

PyTorch implementation of the 3-D Modality-Independent Neighbourhood
Descriptor (MIND) described by Heinrich et al., *MedIA* 2012.

Default parameters reproduce the configuration that gave the lowest
Target Registration Error in the paper (patch 3×3×3, Gaussian σ ≈ 0.5,
six-neighbourhood search region).

Author: Ian Goodall-Halliwell
"""

import math
from typing import Tuple

import torch
from torch import nn


# -----------------------------------------------------------------------------#
#                               MIND descriptor                                 #
# -----------------------------------------------------------------------------#
class MIND3D(nn.Module):
    r"""Modality-Independent Neighbourhood Descriptor for 3-D images.

    Args
    ----
    patch_size : int
        Length of one edge of the cubic patch (must be odd).  The paper
        recommends 3 (i.e. radius 1).
    sigma : float
        Standard deviation of the Gaussian patch weighting (in voxels).
        0.5 is the value used in the reference implementation.
    """

    def __init__(self, patch_size: int = 3, sigma: float = 0.5):
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        self.patch_size = patch_size
        self.sigma2 = sigma ** 2

        # --------------------------- shift kernels --------------------------- #
        # Six one-hot 3×3×3 kernels that pick (+x,-x,+y,-y,+z,-z) neighbours.
        shift_kernels = torch.zeros(6, 1, 3, 3, 3)  # (out_c, in_c, D, H, W)
        centres = [
            (2, 1, 1),  # +z
            (0, 1, 1),  # -z
            (1, 2, 1),  # +y
            (1, 0, 1),  # -y
            (1, 1, 2),  # +x
            (1, 1, 0),  # -x
        ]
        for k, (d, h, w) in enumerate(centres):
            shift_kernels[k, 0, d, h, w] = 1.0

        self.shifter = nn.Conv3d(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=1,
        )
        self.shifter.weight.data.copy_(shift_kernels)
        self.shifter.weight.requires_grad_(False)

        # ----------------------- Gaussian patch kernel ----------------------- #
        g = torch.zeros(1, patch_size, patch_size, patch_size)
        c = (patch_size - 1) / 2.0
        for z in range(patch_size):
            for y in range(patch_size):
                for x in range(patch_size):
                    d2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
                    g[0, z, y, x] = math.exp(-d2 / self.sigma2)
        g /= g.sum()  # normalise so Σ g = 1

        self.patcher = nn.Conv3d(
            in_channels=6,
            out_channels=6,
            kernel_size=patch_size,
            padding=patch_size // 2,
            bias=False,
            groups=6,
        )
        for k in range(6):
            self.patcher.weight.data[k] = g
        self.patcher.weight.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : (B,1,D,H,W) torch.Tensor
            Single-channel 3-D image volumes.

        Returns
        -------
        mind : (B,6,D,H,W) torch.Tensor
            The six-channel MIND descriptor.
        """
        if img.ndim != 5 or img.size(1) != 1:
            raise ValueError("Input must be (B,1,D,H,W)")

        # 1) shift to six neighbours
        shifted = self.shifter(img)  # (B,6,D,H,W)

        # 2) patch-wise squared distances
        diff = shifted - img.repeat_interleave(6, dim=1)
        Dp = self.patcher(diff.pow(2))  # (B,6,D,H,W)

        # 3) local variance V(x): mean over the six channels
        Vx = Dp.mean(dim=1, keepdim=True)  # (B,1,D,H,W)

        # 4) softmax-style normalised descriptor
        num = torch.exp(-Dp / (Vx + 1e-8))
        denom = num.sum(dim=1, keepdim=True)
        mind = num / (denom + 1e-8)
        return mind


# -----------------------------------------------------------------------------#
#                                   Loss                                       #
# -----------------------------------------------------------------------------#
class MINDLoss3D(nn.Module):
    r"""ℓ¹ loss between two 3-D MIND volumes."""

    def __init__(self, patch_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self.descriptor = MIND3D(patch_size, sigma)

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> torch.Tensor:
        """
        moving, fixed : (B,1,D,H,W) tensors.
        Returns scalar loss = mean₍voxels,channels₎ |MINDₘ − MIND𝒻|.
        """
        mind_m = self.descriptor(moving)
        mind_f = self.descriptor(fixed)
        diff = (mind_m - mind_f).abs()  # L1
        B, C, D, H, W = diff.shape
        return diff.sum() / (B * C * D * H * W)


# -----------------------------------------------------------------------------#
#                                 demo / test                                  #
# -----------------------------------------------------------------------------#
def _demo(shape: Tuple[int, int, int] = (64, 64, 64)) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running demo on {device}")

    moving = torch.randn(1, 1, *shape, device=device)
    fixed = torch.randn(1, 1, *shape, device=device)

    criterion = MINDLoss3D().to(device)
    loss = criterion(moving, fixed)
    print(f"MIND-based loss for random volumes: {loss.item():.4f}")


if __name__ == "__main__":
    _demo()
