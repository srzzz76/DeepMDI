import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RangeLoss(nn.Module):
    def __init__(self, lower_bound=0.0, upper_bound=1.0, penalty_weight=10.0, tolerance=0.02):
        """
        RangeLoss with a tolerance margin.
        Penalizes values outside a specific range on the first channel of the input tensor.
        A tolerance margin allows small deviations near the boundaries without incurring a penalty.

        Args:
            lower_bound (float): Minimum allowed value for the first channel.
            upper_bound (float): Maximum allowed value for the first channel.
            penalty_weight (float): Scaling factor for the penalty.
            tolerance (float): Allowed deviation inside the range without penalty.
        """
        super(RangeLoss, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.penalty_weight = penalty_weight
        self.tolerance = tolerance

    def forward(self, x):
        # Apply the range constraint only on the first channel
        channel_1 = x[:, 0, :, :]

        # Define the effective lower and upper bounds considering tolerance
        lower = self.lower_bound - self.tolerance
        upper = self.upper_bound + self.tolerance

        # Penalize values below the lower bound or above the upper bound
        penalty_lower = torch.clamp(lower - channel_1, min=0) ** 2
        penalty_upper = torch.clamp(channel_1 - upper, min=0) ** 2

        penalty = penalty_lower + penalty_upper
        return self.penalty_weight * penalty.sum()


class physics_driven_loss(nn.Module):
    def __init__(self, flag=1):
        super().__init__()
        self.flag = flag
        # Apply a range constraint on the predicted values: expected in [-0.5, 0.5]
        self.range_loss_fn = RangeLoss(lower_bound=-0.5, upper_bound=0.5, penalty_weight=1.0, tolerance=0.02)

    def forward(self, pred, image, circle3):
        """
        Physics-driven loss function for interferometric phase retrieval.
        Combines data fidelity, physical consistency, and range constraints.

        Args:
            pred (Tensor): Network prediction with shape [B, 2, H, W].
                           Channel 0: predicted phase (phi), Channel 1: predicted delta (phase shift).
            image (Tensor): Input interferogram frames with shape [B, 2, H, W].
            circle3 (Tensor): Binary mask defining the valid region.
        """
        # Extract the two input interferogram frames
        frame1 = image[:, 0, :, :]
        frame2 = image[:, 1, :, :]

        # Extract predicted phi and delta
        phi = pred[:, 0, :, :]
        delta = pred[:, 1, :, :]

        # Compute the mean delta over spatial dimensions, scaled to [0, 2π]
        delta_mean = torch.mean(delta, dim=[1, 2]) * 2 * torch.tensor(math.pi)
        delta_mean_expanded = delta_mean.unsqueeze(1).unsqueeze(2).expand_as(phi)

        # Constant amplitude/background term
        AB = torch.tensor(0.5).to(device)
        AB_expanded = AB.expand_as(phi)

        # Wrap phi to [-π, π]
        phi = phi * 2 * math.pi - math.pi

        # Physics-driven loss: enforce consistency with the interferometric model
        # Frame 1: I1 = A + B*cos(phi)
        # Frame 2: I2 = A + B*cos(phi + delta_mean)
        physic1 = AB_expanded + AB_expanded * torch.cos(phi)
        physic2 = AB_expanded + AB_expanded * torch.cos(phi + delta_mean_expanded)

        # Compare reconstructed frames with the original interferograms (within valid region)
        pdloss1 = F.l1_loss(physic1 * circle3, frame1 * circle3)
        pdloss2 = F.l1_loss(physic2 * circle3, frame2 * circle3)

        # Final loss: physics loss + range constraint (scaled)
        loss = pdloss1 + pdloss2 + self.range_loss_fn(pred) * 0.01
        return loss
