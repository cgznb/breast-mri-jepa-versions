#!/usr/bin/env python3
"""CPU default-shape forward check; not a training or medical-quality benchmark."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from response_jepa.config import Config
from response_jepa.contracts import Inputs
from response_jepa.model import ResponseJEPA
from response_jepa.engine import write_json, runtime_info
from response_jepa.provenance import source_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/default_shape_check.json")
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(77)
    model = ResponseJEPA(Config().model)
    model.configure_trainable("C", True)
    model.eval()
    images = torch.zeros(1, 4, 3, 32, 128, 128)
    images[:, 0] = torch.randn(1, 3, 32, 128, 128)
    inp = Inputs(images, torch.tensor([[True, False, False, False]]), torch.tensor([0]),
                 torch.tensor([[0.2, 1., 0., 1.]]), torch.ones(1, 4, dtype=torch.bool),
                 torch.tensor([1]), torch.tensor([0]),
                 torch.tensor([[False, True, True, True]]))
    with torch.no_grad():
        encoded = model.encoder(images[:, 0])
        out = model(inp, need_world=True)
    finite = all(bool(torch.isfinite(v).all()) for v in out.values() if isinstance(v, torch.Tensor))
    assert encoded.shape == (1, 256, 128) and out["states"].shape == (1, 4, 256, 128) and finite
    write_json(a.output, {"scope": "default-shaped synthetic forward only", "environment": runtime_info(),
               "source_identity": source_snapshot(), "input_shape": list(images.shape),
               "encoder_output_shape": list(encoded.shape), "rollout_shape": list(out["states"].shape),
               "all_outputs_finite": finite,
               "parameters_total": sum(x.numel() for x in model.parameters()),
               "parameters_requires_grad_C": sum(x.numel() for x in model.parameters() if x.requires_grad),
               "cuda_tested": False})
    print("Default shape check passed; no real data or training used.")

if __name__ == "__main__":
    main()
