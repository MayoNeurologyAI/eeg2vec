import torch
import pytest

from eeg2vec.pre_process._utils import min_max_normalize
from eeg2vec.pre_process._utils import scaling



def test_min_max_normalize():
    
    # Test 1: Basic functionality of 2D input tensor
    x = torch.Tensor([[0.0, 1.0], [2.0, 3.0]])
    normalized_x, xmax, xmin = min_max_normalize(x)
    assert xmax == 3.0
    assert xmin == 0.0
    # The input tensor x is normalized within [-1, 1] range.
    assert torch.allclose(normalized_x, torch.Tensor([[-1.0000, -0.3333], [0.3333, 1.0000]]), atol=0.0001, rtol=0.0001)
    
    # Test 2: Edge case of 2D input tensor with identical elements
    x = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
    normalized_x, xmax, xmin = min_max_normalize(x)
    # The input tensor x contains identical elements, so both max and min values are the same.
    assert xmax == 1.0
    assert xmin == 1.0
    # Since the elements are identical, the normalized tensor has all zeros.
    assert normalized_x is None



if __name__ == '__main__':
    test_min_max_normalize()