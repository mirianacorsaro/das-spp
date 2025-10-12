import os
import torch
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from .utils.utils import get_first_consecutive_positions

def inference(model, signal, prob_threshold=0.9):
    """
    Runs inference on a signal using our pretrained DAS-SSP model.

    Args:
        model (torch.nn.Module): Pretrained model for segmentation.
        device (torch.device): Computation device (CPU or GPU).
        signal (torch.Tensor): Input signal tensor.
        processing (object): Post-processing module with `post_process_CF_data` method.
        prob_threshold (float, optional): Probability threshold for class selection. Default is 0.8.
    Returns:
        list: A list of detected events with channel, time index, and event type.
    """

    with torch.no_grad(): 
     
        outputs = model(signal)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_mask = torch.argmax(outputs, dim=1)

        binary_masks = {
            "P": ((predicted_mask == 1) & (probs[:, 1] > prob_threshold)).cpu().float(),
            "S": ((predicted_mask == 2) & (probs[:, 2] > prob_threshold)).cpu().float(),
        }

        events = {1 : get_first_consecutive_positions(binary_masks["P"].squeeze(0), 1),
                  2 : get_first_consecutive_positions(binary_masks["S"].squeeze(0), 1)}

        results = [
            [x, y, event_type]
            for event_type, (x_positions, y_positions) in events.items()
            for x, y in zip(x_positions, y_positions)
        ]

        picks = np.array(results, dtype=np.int32)
        
    return picks
