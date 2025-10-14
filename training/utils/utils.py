import numpy as np
import torch

def get_first_consecutive_positions(mask: np.ndarray, value: int):
    """
    Finds the first positions of consecutive occurrences of a given value in a binary mask.

    Args:
        mask (np.ndarray): The binary mask.
        value (int): The value to search for.

    Returns:
        list: x-coordinates (channels)
        list: y-coordinates (time indices)
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected [H,W] or [1,H,W], got {tuple(mask.shape)}")

    if mask.dtype is not torch.bool:
        mask = (mask == value)

    mask = mask.to("cpu", non_blocking=True)

    prev = torch.zeros_like(mask, dtype=torch.bool)
    prev[:, 1:] = mask[:, :-1]

    starts = mask & (~prev)
    x, y = torch.nonzero(starts, as_tuple=True)
    return x.tolist(), y.tolist()

def calculate_accuracy(inputs, outputs):
    inputs = inputs.flatten(1)
    outputs = outputs.flatten(1)
    numerator = 2 * (inputs * outputs).sum(-1)
    denominator = inputs.sum(-1) + outputs.sum(-1)
    dice = (numerator + 1) / (denominator + 1)
    return dice.mean()

def compute_dice_loss(inputs, outputs):
    inputs = inputs.flatten(1)
    outputs = outputs.flatten(1)
    num_masks = len(inputs)
    numerator = 2 * (inputs * outputs).sum(-1)
    denominator = inputs.sum(-1) + outputs.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks
