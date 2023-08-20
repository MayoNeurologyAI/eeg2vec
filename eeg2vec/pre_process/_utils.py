import re
import mne
import torch
import numpy as np
from typing import Union, Tuple
from torch.nn.functional import interpolate


def min_max_normalize(x: torch.Tensor, low=-1, high=1) -> Tuple[torch.Tensor, float, float]:
    """ 
    This function performs min-max normalization
    
    Parameters
    ----------
    x : torch.Tensor
        EEG Epoch
    
    low : int, default=-1
        Lower bound of the normalization
    
    high : int, default=1
        Upper bound of the normalization
    
    Returns
    -------
    x : torch.Tensor
        Normalized EEG Epoch
    
    xmax : float
        Maximum value of the EEG Epoch
    
    xmin : float
        Minimum value of the EEG Epoch
    
    """
    
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = None
            return x, xmax, xmin
        
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    
    return (high - low) * x, xmax, xmin
    
def scaling(x: torch.Tensor, data_max: float, data_min: float) -> float:
    """ 
    This function returns the constant value for adding scaling channel
    
    Parameters
    ----------
    x : torch.Tensor
        EEG Epoch
    
    data_max : float
        Maximum value of the EEG Epoch
    
    data_min : float
        Minimum value of the EEG Epoch
    
    Returns
    -------
    scale : float
        Constant value for adding scaling channel
        
    """
    tol = 0.0001
    
    max_scale = data_max - data_min
    
    p90 = torch.quantile(x, 0.9, dim=-1)
    p10 = torch.quantile(x, 0.1, dim=-1)
    
    min_, max_ = torch.amin(p10), torch.amax(p90)

    # min_, max_ = x.min(), x.max()
    
    if (max_ > (data_max + tol)) or (min_ < (data_min - tol)):
        print(f" Max data: {max_:.4} & Min data: {min_:.4}")
        return None
    
    scale = 2 * (torch.clamp_max((max_ - min_) / max_scale, 1.0) - 0.5)
    
    return scale

def normalize_scale_interpolate(x: Union[np.ndarray, torch.Tensor], 
                                sequence_len: int = 6,
                                new_sfreq: int =256,
                                dataset_max: float = 0.001,
                                dataset_min: float = -0.001):
    """ 
    This function performs normalization, scaling, adding a stim channel, 
    and temporal interpolation
    
    Parameters
    ----------
    x : torch.Tensor/ np.ndarray
        EEG data
    
    sequence_len : int, default=6
        Length of the sequence in seconds
    
    new_sfreq : int, default=256
        New sampling frequency
        
    dataset_max: float
        Maximum value in the dataset
        
    dataset_min: float
        Minimum value in the dataset
    
    Returns
    -------
    x : torch.Tensor
        Normalized, scaled, and interpolated EEG data
    
    """
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    #scaling
    constant_value = scaling(x, dataset_max, dataset_min)
    
    if constant_value is None:
        return None
    
    # Normalization
    x, xmax, xmin = min_max_normalize(x)
    
    if x is None:
        print ("Normalization x value is 0")
        return None
    
    # Temporal interpolation
    new_sequence_length = int(sequence_len * new_sfreq)
    if not (x.shape[1] == new_sequence_length):
        x = interpolate(x.unsqueeze(0), new_sequence_length, mode="nearest").squeeze(0)
        
    # add new channel
    # Create a new row filled with the constant value
    new_row = torch.full((1, x.shape[1]), constant_value)
    x = torch.cat((x, new_row), dim=0)
    
    return x

def annotate_nans(eeg):
    """
    Add NAN annotations to already existing annotations

    Parameters
    ----------

    Returns
    -------
    None
    """
    # Annotate NAN's
    nan_annot = mne.preprocessing.annotate_nan(eeg)
    update_annot = eeg.annotations + nan_annot
    eeg.set_annotations(update_annot)
    return eeg

