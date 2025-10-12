import numpy as np

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
    mask = np.array(mask) 
    shifted_mask = np.roll(mask, shift=1, axis=1) 
    shifted_mask[:, 0] = 0 
    condition = (mask == value) & (shifted_mask != value)  
    x_positions, y_positions = np.where(condition) 
    return x_positions.tolist(), y_positions.tolist()

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
