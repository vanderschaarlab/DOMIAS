# stdlib
import random

# third party
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def run_before() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
