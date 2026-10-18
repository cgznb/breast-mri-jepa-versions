from dataclasses import replace
import pytest
import torch
from response_jepa.synthetic import small_config, make_cohort
from response_jepa.model import ResponseJEPA
from response_jepa.contracts import Inputs, Targets, Batch

@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.set_num_threads(2)
    torch.manual_seed(1234)

@pytest.fixture
def config():
    return small_config("C")

@pytest.fixture
def model(config):
    m = ResponseJEPA(config.model)
    m.configure_trainable("C", True)
    return m.eval()

@pytest.fixture
def batch():
    images = torch.randn(2, 4, 3, 8, 16, 16)
    observed = torch.tensor([[True, False, False, False], [True, True, False, False]])
    inp = Inputs(images.masked_fill(~observed[:, :, None, None, None, None], 0), observed,
                 torch.tensor([0, 1]), torch.tensor([[0.3, 1, 0, 1], [-.1, 0, 1, 1]]),
                 torch.ones(2, 4, dtype=torch.bool), torch.tensor([1, 2]), torch.tensor([0, 0]),
                 torch.tensor([[False, True, True, True], [False, False, True, True]]))
    sup = Targets(images, ~observed, torch.tensor([0., 1.]), torch.ones(2, dtype=torch.bool))
    return Batch(inp, sup, ("p0", "p1"))

@pytest.fixture
def cohort(tmp_path):
    return make_cohort(tmp_path/"data", counts=(8, 4, 4))
